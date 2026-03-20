# Change: Move inference batches to the accelerator device

Date: 2026-03-12

## Summary
Updated the inference helpers to move incoming batches onto `accelerator.device` before the model forward.

## Motivation
Evaluation batches were staying on CPU because the test dataloader was not wrapped by `accelerator.prepare`, which caused device mismatches once the model weights were on CUDA.

## Files Modified
- src/dust3r/inference.py
- changes/README.md

## Notes
This keeps training and evaluation consistent even when only the training dataloader is prepared through Accelerate.
