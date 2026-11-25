"""
Tests for tpv.csp (covariance matrices, CSP filters, transform shape).
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")) # noqa: E402
import scipy

from tpv.data import load_raw
from tpv.preprocessing import get_xy
from tpv.csp import CSP


def test_csp_fit() -> None:
    X, y = get_xy(load_raw(4, 14))
    print(CSP(4).fit(X, y).filters_.shape)


def test_eigval() -> None:
    """
    固有値分解
    """
    A = np.array([[3, 1], [1, 3]])
    eigvals, eigvecs = np.linalg.eigh(A)
    print(eigvals, eigvecs)

    print(f"{'=' * 10}固有値チェック i = 0{'='* 10}")
    i = 0
    print(A @ eigvecs[:, i])
    print(eigvals[i] * eigvecs[:, i])

    print(f"{'=' * 10}固有値チェック i = 1{'='* 10}")
    i = 1
    print(A @ eigvecs[:, i])
    print(eigvals[i] * eigvecs[:, i])


def test_real_eigval() -> None:
    """
    実データ
    """
    X, y = get_xy(load_raw(4, 14))
    csp = CSP(4)
    C1 = csp._class_covariance(X, y, 1)
    C2 = csp._class_covariance(X, y, 2)

    eigvals, eigvecs = scipy.linalg.eigh(C1, C1 + C2)
    print(eigvals.shape)
    print(eigvecs.shape)
    print(eigvals.min())
    print(eigvals.max())

    print(f"{'=' * 10}固有値チェック i = 0{'='* 10}")
    i = 0
    w = eigvecs[:, i]
    λ = eigvals[i]
    # print(C1 @ w)
    # print(λ * (C1 + C2) @ w)
    print(np.allclose(C1 @ w, λ * (C1 + C2) @ w))

    print(f"{'=' * 10}固有値チェック i = 1{'='* 10}")
    i = 1
    w = eigvecs[:, i]
    λ = eigvals[i]
    # print(C1 @ w)
    # print(λ * (C1 + C2) @ w)
    print(np.allclose(C1 @ w, λ * (C1 + C2) @ w))


def test_cov_matrix() -> None:
    """
    分散共分散行列テスト
    """
    X = np.array([[2, -1, 1], [2, -2, 1]])

    print(f"{'=' * 10}元の形状{'=' * 10}")
    print(X.shape)

    print(f"{'=' * 10}共分散行列{'=' * 10}")
    C = X @ X.T
    print(C)

    print(f"{'=' * 10}対角和{'=' * 10}")
    print(np.trace(C))

    print(f"{'=' * 10}正規化後の対角和(=1){'=' * 10}")
    print(np.trace(C / np.trace(C)))


def test_real_epoch() -> None:
    """
    実際のエポック
    """
    raw = load_raw(4, 14)
    X, y = get_xy(raw)
    print(X.shape)
    # print(y)

    print(f"{'=' * 10}次元を揃える{'=' * 10}")
    E = X[0]
    print(E.shape)
    C = E @ E.T
    print(C)

    print(f"{'=' * 10}正規化後の対角和(=1){'=' * 10}")
    print(np.trace(C / np.trace(C)))

    print(f"{'=' * 10}CSP.pyテスト{'=' * 10}")
    print(np.trace(CSP._covariance(X[0])))


if __name__ == "__main__":
    try:
        test_cov_matrix()
        print(f"{'=' * 10}リアル分散共分散行列{'='* 10}")
        test_real_epoch()

        print(f"{'=' * 10}固有値分解{'='* 10}")
        test_eigval()

        print(f"{'=' * 10}リアル固有値分解{'='* 10}")
        test_real_eigval()

        print(f"{'=' * 10}CSP fit test{'='* 10}")
        test_csp_fit()

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
