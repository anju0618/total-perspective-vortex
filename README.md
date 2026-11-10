# total-perspective-vortex

42 project: a brain computer interface. Classify motor imagery EEG
(hand or feet movement) with MNE and scikit-learn.

## Layout

| Path | Content |
|---|---|
| `mybci.py` | entry point (`train`, `predict`, or no argument for the evaluation of all subjects) |
| `src/visualize.py` | raw EEG and filtered EEG plots |
| `src/tpv/data.py` | loading the physionet data with MNE |
| `src/tpv/preprocessing.py` | filtering and feature extraction |
| `src/tpv/csp.py` | own dimensionality reduction (sklearn transformer) |
| `src/tpv/pipeline.py` | sklearn `Pipeline` |
| `src/tpv/stream.py` | playback of a recording as a stream |

## Usage

```
make install
make visualize SUBJECT=4 RUN=14
make train SUBJECT=4 RUN=14
make predict SUBJECT=4 RUN=14
make eval
```

`make help` lists everything.
