from pymystem3 import Mystem
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ["TextLemmatizer"]


class TextLemmatizer(BaseEstimator, TransformerMixin):
    stemmer: Mystem

    def __init__(self):
        if (
            not hasattr(TextLemmatizer, "stemmer")
            or getattr(TextLemmatizer, "stemmer") is None
        ):
            TextLemmatizer.stemmer = Mystem()
        self.stemmer = TextLemmatizer.stemmer

    def stem(self, string):
        return "".join(self.stemmer.lemmatize(string))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for x in X:
            yield self.stem(x)
