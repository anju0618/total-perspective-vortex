"""Load EEG of the physionet motor imagery dataset with MNE."""

import os
from pathlib import Path
from dotenv import load_dotenv

import mne
from mne.datasets import eegbci

import matplotlib.pyplot as plt

load_dotenv()
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))


def fetch(subject: int, runs: list[int]) -> list[str]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return eegbci.load_data(subject, runs, path=DATA_DIR, update_path=False)


def read_mne():
    paths = fetch(4, [14])
    raw = mne.io.read_raw_edf(paths[0], preload=True)
    print(raw.info)
    raw.plot()
    plt.show()


if __name__ == "__main__":
    read_mne()