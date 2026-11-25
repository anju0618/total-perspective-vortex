"""
複数のチャンネルを足して、合わせる。
a, bの2ch
両手：AもBも大きく動く
足: AもBも小さい

線形代数
z = a + b: 共通している動きが強まる
z = a - b: 共通の動きが打ち消され、違いのみが抽出
この足し方に重み付けウェイトをつけるw
z = w1*A + w2*B
w = (1, 1)が足す
w = (1, -1)が引く

"""
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from scipy import linalg


class CSP(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """
    CSPするクラス
    """
    def __init__(self, n_components: int = 4) -> None:
        if n_components % 2 != 0 or n_components < 2:
            raise ValueError("n_components should be positive even num")

        self.n_components = n_components

    @staticmethod
    def _covariance(E: np.ndarray) -> np.ndarray:
        """
        行列共分散
        """
        C = E @ E.T
        return C / np.trace(C)

    def _class_covariance(self, X: np.ndarray, y: np.ndarray,
                          label: int) -> np.ndarray:
        """
        クラスごとのCov matrix mean
        """
        X = X[y == label]
        res = []
        for x in X:
            res.append(self._covariance(x))

        return np.mean(res, axis=0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CSP":
        if self.n_components % 2 != 0:
            raise ValueError("n_components should be even number")

        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("CSP needs exactly 2 classes")

        C = []
        for i in classes:
            Ci = self._class_covariance(X, y, i)
            C.append(Ci)

        eigvals, eigvecs = linalg.eigh(C[0], C[0] + C[1])

        half = self.n_components // 2
        min_w = eigvecs[:, :half]
        max_w = eigvecs[:, -half:]
        self.filters_ = np.hstack([min_w, max_w]).T
        return self

    def transform(self, X):  # type: ignore[no-untyped-def]
        raise NotImplementedError
