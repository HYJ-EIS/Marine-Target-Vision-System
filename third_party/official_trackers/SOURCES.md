# Official Tracker Sources

This directory vendors selected tracker modules so the project can compare its
existing lightweight baseline with official tracker implementations on the same
detector outputs.

## OC-SORT

- Source: https://github.com/noahcao/OC_SORT
- Vendored commit: `8462e7e729a93ccd3bd995c0a79a890336cb3a0b`
- License copy: `OC_SORT_LICENSE`
- Vendored files: `oc_sort/ocsort_tracker/{ocsort.py,association.py,kalmanfilter.py}`
- Local compatibility changes: package-relative imports are preserved; `kalmanfilter.py` has a local fallback for small `filterpy` helpers when `filterpy` is not installed.

## BoT-SORT

- Source: https://github.com/NirAharon/BoT-SORT
- Vendored commit: `251985436d6712aaf682aaaf5f71edb4987224bd`
- License copy: `BOT_SORT_LICENSE`
- Vendored files: `bot_sort/tracker/{bot_sort.py,matching.py,kalman_filter.py,basetrack.py,gmc.py}`
- Local compatibility changes: imports are package-relative, ReID remains disabled unless original FastReID dependencies are available, `lap` and `cython_bbox` have SciPy/NumPy fallbacks, and sparse optical-flow GMC returns identity on featureless frames.

## Project adapter

- Adapter: `target_module/image_detect_module/utils/official_adapter.py`
- Tracker names: `official_ocsort`, `official_botsort`
- Output metrics from project experiment scripts are proxy engineering metrics, not MOTChallenge IDF1/HOTA/MOTA scores.
