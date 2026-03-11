# Change: Add bayesian init checkpoint script

Date: 2026-03-11

## Summary
Added a helper script that converts an existing CUT3R checkpoint into a Bayesian decoder initialization checkpoint by injecting the new model configuration fields and loading the original weights with `strict=False`.

## Motivation
The demo and future fine-tuning runs need a reproducible way to start from the standard `cut3r_512_dpt_4_64.pth` checkpoint while initializing the new decoder-side Bayesian modules.

## Files Modified
- scripts/make_bayesian_decoder_init_ckpt.py
- changes/README.md

## Notes
The generated checkpoint stays outside the repository under `../checkpoints` and is intentionally not committed.
