# Change: Document Bayesian CUT3R decoder PoC

Date: 2026-03-11

## Summary
Documented how the current `bayes_cut3r` worktree differs from the original CUT3R repository and summarized the decoder-only Bayesian adaptation PoC.

## Motivation
The repository needs a traceable record of the current Bayesian decoder design, training changes, and dataset/config additions so later experiments can start from a clear baseline.

## Files Modified
- src/dust3r/blocks.py
- src/dust3r/model.py
- src/dust3r/inference.py
- src/train.py
- src/dust3r/datasets/kitti_odometry.py
- src/dust3r/datasets/__init__.py
- config/bayes_kitti_odometry_poc.yaml

## Notes
The PoC keeps the encoder and downstream head structurally unchanged, injects state-conditioned layer-wise alpha values into the recurrent decoder, and adds KL regularization based on the decoder weight counts controlled by each alpha.
