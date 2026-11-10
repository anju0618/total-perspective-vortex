"""Own dimensionality reduction (CSP, or PCA...).

Must be a sklearn transformer (BaseEstimator + TransformerMixin).
"""
from sklearn.base import BaseEstimator, TransformerMixin


class CSP(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    def fit(self, X, y):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def transform(self, X):  # type: ignore[no-untyped-def]
        raise NotImplementedError
