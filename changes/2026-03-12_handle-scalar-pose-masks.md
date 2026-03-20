# Change: Handle scalar pose masks in evaluation loss

Date: 2026-03-12

## Summary
Updated pose loss computation so scalar `pose_masks` from batch-size-1 evaluation runs are handled as all-true or all-false masks instead of being iterated as sequences.

## Motivation
The KITTI PoC evaluation hit `TypeError: iteration over a 0-d tensor` inside `compute_pose_loss()` when `pose_masks` collapsed to a scalar tensor.

## Files Modified
- src/dust3r/losses.py
- changes/README.md

## Notes
Zero-mask cases now return a zero tensor on the same device as the predictions.
