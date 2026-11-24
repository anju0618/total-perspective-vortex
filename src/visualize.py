"""Plot raw EEG, then the same data after filtering.

python src/visualize.py <subject> <run>

Figures go to FIG_DIR/S<subject>R<run>_raw/ and
FIG_DIR/S<subject>R<run>_filtered/
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tpv.data import load_raw, save_figs  # noqa: E402
from tpv.preprocessing import filter_raw  # noqa: E402


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


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
