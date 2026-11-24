"""フィルタと特徴の取り出し（周波数帯、エポック）"""

import sys

import mne
import numpy as np

# 運動イメージのとき、μ波（8〜13 Hz）とβ波（13〜30 Hz）の強さが変わる
L_FREQ = 8.0
H_FREQ = 30.0
# T1 / T2 が2つのクラス（run 14：両拳 / 両足）
# T0（休み）は分類には使わない
EVENT_ID = {"T1": 1, "T2": 2}
EPOCH_LENGTH = 4.0  # 秒（1つの合図は 4.1 秒続く）


def filter_raw(raw: mne.io.BaseRaw,
               l_freq: float = L_FREQ,
               h_freq: float = H_FREQ) -> mne.io.BaseRaw:
    """
    `raw` のコピーにバンドパスフィルタをかける（元の `raw` は変えない）
    """
    return raw.copy().filter(l_freq, h_freq, verbose=False)


def make_epochs(raw: mne.io.BaseRaw) -> mne.Epochs:
    """
    T1 / T2 の印ごとに、エポックを1つ切り出す
    """
    events, _ = mne.events_from_annotations(
        raw, event_id=EVENT_ID, verbose=False)
    return mne.Epochs(raw, events, event_id=EVENT_ID,
                      tmin=0.0, tmax=EPOCH_LENGTH, baseline=None,
                      preload=True, verbose=False)


def get_xy(raw: mne.io.BaseRaw) -> tuple[np.ndarray, np.ndarray]:
    """
    フィルタ → 切り出し → X（エポック, チャンネル, 時間）と y（1 か 2）を返す
    """
    epochs = make_epochs(filter_raw(raw))
    return epochs.get_data(copy=True), epochs.events[:, 2]


def band_ratio(raw: mne.io.BaseRaw,
               l_freq: float = L_FREQ,
               h_freq: float = H_FREQ) -> float:
    """帯域の中のパワー / 帯域の外のパワー（フィルタが効いたかの目安）"""
    psd = raw.compute_psd(fmin=1.0, fmax=70.0, verbose=False)
    power = psd.get_data().mean(axis=0)  # チャンネルの平均
    inside = (psd.freqs >= l_freq) & (psd.freqs <= h_freq)
    return float(power[inside].sum() / power[~inside].sum())


def main() -> None:
    """チェック用：python src/tpv/preprocessing.py <subject> <run>"""
    from collections import Counter
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tpv.data import load_raw

    subject = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    run = int(sys.argv[2]) if len(sys.argv) > 2 else 14

    print(f"{'=' * 20} load subject {subject}, run {run} {'=' * 20}")
    raw = load_raw(subject, run)
    print(f"channels: {len(raw.ch_names)}, "
          f"sfreq: {raw.info['sfreq']} Hz, "
          f"length: {raw.times[-1]:.1f} s")
    print(f"annotations: {raw.annotations}")

    print(f"\n{'=' * 20} filter {L_FREQ}-{H_FREQ} Hz {'=' * 20}")
    filtered = filter_raw(raw)
    print(f"in-band / out-of-band power  raw: {band_ratio(raw):.3f}  "
          f"filtered: {band_ratio(filtered):.3f}")
    assert band_ratio(filtered) > band_ratio(raw), "filter had no effect"

    print(f"\n{'=' * 20} epochs {'=' * 20}")
    epochs = make_epochs(filtered)
    print(epochs)

    print(f"\n{'=' * 20} get_xy {'=' * 20}")
    X, y = get_xy(raw)
    print(f"X shape: {X.shape}  (epochs, channels, times)")
    print(f"y shape: {y.shape}, classes: {dict(sorted(Counter(y).items()))}")
    assert X.shape[0] == y.shape[0], "X and y have different lengths"
    assert set(y) <= set(EVENT_ID.values()), "unexpected label in y"
    assert not np.isnan(X).any(), "X has NaN"
    print("OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
