# Change: Split KITTI PoC train and eval sequences

Date: 2026-03-12

## Summary
Updated the KITTI odometry PoC config to train on sequences `01` through `10` and evaluate on sequence `00`.

## Motivation
This PoC is intended to measure whether lightweight Bayesian decoder fine-tuning on held-out KITTI odometry data improves performance on sequence `00`.

## Files Modified
- config/bayes_kitti_odometry_poc.yaml
- changes/README.md

## Notes
The experiment name was updated so checkpoints and logs for this split are clearly separated from the earlier all-in-one PoC config.
