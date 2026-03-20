# Change: Control auto-resume and mixed precision

Date: 2026-03-12

## Summary
Made `train.py` respect explicit auto-resume and mixed-precision settings from the config, and updated the KITTI PoC config to disable auto-resume and `Accelerator` mixed precision by default.

## Motivation
The PoC smoke runs were unintentionally picking up `checkpoint-last.pth` even when `resume=null` was passed, and evaluation was still seeing dtype issues because `Accelerator` was hard-coded to `bf16`.

## Files Modified
- src/train.py
- config/bayes_kitti_odometry_poc.yaml
- changes/README.md

## Notes
Training AMP remains controlled by `amp`, while `mixed_precision` now only affects the `Accelerator` wrapper. The PoC config uses `mixed_precision='no'` so evaluation stays in full precision by default.
