from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

__all__ = ["clusters_with_tfidf"]


def clusters_with_tfidf(docs_all):
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(docs_all)
    clusterer = DBSCAN(eps=0.2, min_samples=2, metric='euclidean')
    cluster_labels = clusterer.fit_predict(X_tfidf)
    return cluster_labels