import re
import unicodedata

import nltk
from bs4 import BeautifulSoup
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ["TextNormalizer"]


class TextNormalizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        language="russian",
        lower=True,
        strip_characters=False,
        stopwords=False,
        remove_tags=False,
        fix_text=True,
    ):
        self.language = language
        self.stopwords = set(nltk.corpus.stopwords.words(language))
        self._lower = lower
        self._strip_characters = strip_characters
        self._stopwords = stopwords
        self._remove_tags = remove_tags
        self._fix_text = fix_text

    def is_punct(self, token):
        return all(unicodedata.category(char).startswith("P") for char in token)

    def is_stopword(self, token):
        return token.lower() in self.stopwords

    def strip_characters(self, item):
        # item = re.sub(r';', '.', item)
        item = re.sub(r"[0-9]+\.", "", item)
        item = re.sub(r"[а-яА-Я]\)", "", item)
        item = re.sub(r"- ", " ", item)
        item = re.sub(r"^-", "", item)
        item = re.sub(r"[;\.–·]$", "", item)

        output = ""
        if len(item) == 0:
            return output
        for character in item:
            if character.isalnum() or character in [
                "\n",
                " ",
                "-",
                ",",
                "(",
                ")",
                "/",
                "\\",
                "!",
                "?",
                ";",
                ".",
            ]:
                if character == "\n" and len(output) > 0 and output[-1] != ".":
                    output += "."
                output += character

        return output

    def remove_tags(self, string):
        soup = BeautifulSoup(string, features="lxml")
        string = soup.get_text("\n")
        return string

    def fix_text(self, text):
        text = text.replace("\n", " ")
        text = re.sub(r"\.+", ".", text)
        text = re.sub("\t", " ", text)
        text = re.sub(r"( )+", " ", text)
        text = re.sub(r" \.", ".", text)
        text = re.sub(r"\.{1,} ", ". ", text)
        return text

    def normalize(self, string):
        string = string.strip()
        if self._remove_tags:
            string = self.remove_tags(string)
        if self._lower:
            string = string.lower()
        if self._strip_characters:
            string = self.strip_characters(string)
        if self._fix_text:
            string = self.fix_text(string)
        if self._stopwords:
            string = " ".join(
                [
                    token
                    for token in nltk.tokenize.word_tokenize(string)
                    if not self.is_stopword(token) and not self.is_punct(token)
                ]
            )
        return string

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for x in X:
            yield self.normalize(x)
