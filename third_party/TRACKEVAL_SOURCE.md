# TrackEval Source

- Source: https://github.com/JonathonLuiten/TrackEval
- Vendored commit: `12c8791b303e0a0b50f753af204249e622d0281a`
- License copy: `third_party/TRACKEVAL_LICENSE`
- Purpose: official MOTChallenge-style HOTA, CLEAR/MOTA, and Identity/IDF1 evaluation.
- Local wrapper: `tools/evaluation/motchallenge_eval.py`

The project wrapper expects MOTChallenge-format ground truth and tracker result files. These metrics require real cross-frame ground-truth identities; they cannot be derived from tracker output alone.
