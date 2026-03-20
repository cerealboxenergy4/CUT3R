# Change: Fix demo KITTI sequence input handling

Date: 2026-03-11

## Summary
Updated `demo.py` so it defaults to the Bayesian decoder initialization checkpoint and accepts KITTI odometry sequence folders directly.

## Motivation
The demo should run against the Bayesian decoder checkpoint without extra flags, and it should handle the local KITTI directory layout without requiring manual selection of `image_2`.

## Files Modified
- demo.py
- .gitignore
- changes/README.md

## Notes
`parse_seq_path()` now descends into `image_2` or `image_3` when a sequence root is provided and also corrects the common `images_2` or `images_3` typo when those folders are passed explicitly.
