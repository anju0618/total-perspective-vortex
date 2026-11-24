"""フィルタと特徴の取り出し（周波数帯、エポック）"""

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
