# Change: Align environment for gsplat support

Date: 2026-03-12

## Summary
Replaced the old Conda environment spec with an `environment.yml` file that creates a `bayes_cut3r` environment pinned to a `torch 2.4 + CUDA 12.4` combination compatible with the official `gsplat` wheel.

## Motivation
The existing `cut3r_124` environment uses `torch 2.5.1`, which is outside the documented `gsplat` wheel support range and caused the training smoke test to fail before the first step.

## Files Modified
- environment.yml
- requirements.yml
- changes/README.md

## Notes
The new spec removes local CUDA toolkit and `nvcc` pins so `gsplat` can use the prebuilt `pt24cu124` wheel instead of relying on a local source build.
