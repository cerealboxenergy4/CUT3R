# Change: Fix KITTI resolution handling in Hydra config

Date: 2026-03-12

## Summary
Inlined the KITTI training resolution list inside the dataset expression so Hydra passes it through `eval()` in the same format used by the original CUT3R configs.

## Motivation
The previous `${kitti_resolution}` interpolation was being stringified into token fragments such as `['(512', '384)', ...]`, which caused the training smoke test to fail before the first batch.

## Files Modified
- config/bayes_kitti_odometry_poc.yaml
- changes/README.md

## Notes
The training smoke test now reaches model construction and checkpoint loading instead of failing during dataset initialization.
