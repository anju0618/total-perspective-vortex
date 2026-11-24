"""physionet の運動イメージの脳波を、MNE で読み込む"""

import os
from pathlib import Path
from dotenv import load_dotenv

import mne
from mne.datasets import eegbci

import matplotlib.pyplot as plt

load_dotenv()
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
FIG_DIR = Path(os.environ.get("FIG_DIR", "figures"))


def fetch(subject: int, runs: list[int]) -> list[str]:
    """
    subject: 1~109
    runs: 1~14
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return eegbci.load_data(subject, runs, path=DATA_DIR, update_path=False)


def load_raw(subject: int, run: int) -> mne.io.BaseRaw:
    """記録を1つ取ってきて、Raw オブジェクトとして読む"""
    path = fetch(subject, [run])[0]
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    eegbci.standardize(raw)  # "C3.." → "C3"（標準のチャンネル名にそろえる）
    return raw


def save_figs(raw: mne.io.BaseRaw, name: str, width: int = 30) -> None:
    """波形を `width` 秒ずつに区切って、図を何枚かに分けて保存する

    保存先：FIG_DIR/name/name_000.png, name_030.png, ...
    """
    out = FIG_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    total = raw.times[-1]
    for start in range(0, int(total), width):
        fig = raw.plot(start=start, duration=min(width, total - start),
                       n_channels=len(raw.ch_names), show=False)
        fig.set_size_inches(20, 12)
        fig.savefig(out / f"{name}_{start:03d}.png")
        plt.close(fig)


def read_mne():

    x = 0
    print(f"{'=' * 20} fetching raw data {'=' * 20}")
    paths = fetch(1, [14])
    raw = mne.io.read_raw_edf(paths[x], preload=True)

    print(f"\n\n{'=' * 20} print raw annotation {'=' * 20}")
    print(raw.annotations)

    print(f"\n\n{'=' * 10} loading {paths[x]} data {'=' * 10}")
    print(raw.info)

    print(f"\n\n{'=' * 10} printing psd {'=' * 10}")
    raw.compute_psd().plot()
    save_figs(raw, Path(paths[x]).stem)


if __name__ == "__main__":
    read_mne()
