"""Entry point.

python mybci.py <subject> <run> train     train and show cross_val_score
python mybci.py <subject> <run> predict   play back the file as a stream
python mybci.py                           mean accuracy, all subjects
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
