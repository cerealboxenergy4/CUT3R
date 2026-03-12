# Change: Disable AMP during evaluation forward

Date: 2026-03-12

## Summary
Updated the inference helpers so CUDA autocast is only enabled when mixed precision is requested and the model is in training mode.

## Motivation
The KITTI Bayesian PoC could train, but evaluation reused the training autocast path and hit mixed-precision dtype mismatches in the encoder.

## Files Modified
- src/dust3r/inference.py
- changes/README.md

## Notes
Training still uses AMP when `amp=1`, while validation and other evaluation-style forwards now run in full precision by default.
