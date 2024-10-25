import nltk
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ["TextSnowballStemmer"]


class TextSnowballStemmer(BaseEstimator, TransformerMixin):
    def __init__(self, language="russian"):
        self.language = language
        self.stemmer = nltk.stem.SnowballStemmer(language)

    def stem(self, string):
        return " ".join(
            [self.stemmer.stem(token) for token in nltk.tokenize.word_tokenize(string)]
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for x in X:
            yield self.stem(x)
