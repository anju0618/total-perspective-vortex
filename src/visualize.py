"""Plot raw EEG, then the same data after filtering.

python src/visualize.py <subject> <run>

Figures go to FIG_DIR/S<subject>R<run>_raw/ and
FIG_DIR/S<subject>R<run>_filtered/. It also prints the matrix X
(epochs, channels, times) and saves FIG_DIR/S<subject>R<run>_matrix.png.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402

from tpv.data import FIG_DIR, load_raw, save_figs  # noqa: E402
from tpv.preprocessing import filter_raw, get_xy, make_epochs  # noqa: E402


def show_matrix(raw: mne.io.BaseRaw, name: str,
                n_ch: int = 5, n_t: int = 8) -> None:
    """エポック 0 の行列（チャンネル × 時間）を、フィルタ前後で表示する

    数字は µV。1つの行列は 64 × 641 で大きいので、左上の一部だけ表示し、
    全体はヒートマップに保存する。
    """
    X_raw = make_epochs(raw).get_data(copy=True) * 1e6
    X, y = get_xy(raw)
    X = X * 1e6
    print(f"\nX shape: {X.shape}  (epochs, channels, times)")
    print(f"y: {y}")

    np.set_printoptions(precision=2, suppress=True, linewidth=150)
    for label, data in (("raw", X_raw), ("filtered", X)):
        print(f"\n{label}: epoch 0, first {n_ch} channels x "
              f"first {n_t} samples [µV]")
        for i in range(n_ch):
            print(f"{raw.ch_names[i]:>4}", data[0, i, :n_t])
        print(f"min {data[0].min():.1f}  max {data[0].max():.1f}  "
              f"std {data[0].std():.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, label, data in zip(axes, ("raw", "filtered"), (X_raw, X)):
        lim = np.percentile(np.abs(data[0]), 99)
        im = ax.imshow(data[0], aspect="auto", cmap="RdBu_r",
                       vmin=-lim, vmax=lim)
        ax.set_title(f"{label}: epoch 0 (y={y[0]})")
        ax.set_xlabel("time (sample)")
        ax.set_ylabel("channel")
        fig.colorbar(im, ax=ax, label="µV")
    out = FIG_DIR / f"{name}_matrix.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"\nmatrix   -> {out}")


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    subject, run = int(sys.argv[1]), int(sys.argv[2])

    raw = load_raw(subject, run)
    name = f"S{subject:03d}R{run:02d}"

    print(f"raw      -> figures/{name}_raw")
    save_figs(raw, f"{name}_raw")
    print(f"filtered -> figures/{name}_filtered")
    save_figs(filter_raw(raw), f"{name}_filtered")
    show_matrix(raw, name)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
