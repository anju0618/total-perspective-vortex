"""Load EEG of the physionet motor imagery dataset with MNE."""

import os
from pathlib import pathlib
from dotenv import load_dotenv

import mne
from mne.datasets import eegbci

load_dotenv()
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))