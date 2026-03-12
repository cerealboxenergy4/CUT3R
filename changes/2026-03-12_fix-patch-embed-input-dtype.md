# Change: Fix patch embed input dtype

Date: 2026-03-12

## Summary
Cast image inputs to the patch embedding convolution dtype before applying the projection in both patch embed implementations.

## Motivation
The evaluation pass of the KITTI Bayesian PoC hit a dtype mismatch where the image tensor stayed in `float32` while the patch embedding bias had been cast to `bfloat16`.

## Files Modified
- src/dust3r/patch_embed.py
- changes/README.md

## Notes
This keeps the patch embedding input aligned with the model dtype in both training and evaluation, including mixed-precision runs.
