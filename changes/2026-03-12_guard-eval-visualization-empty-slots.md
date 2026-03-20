# Change: Guard evaluation visualization against empty slots

Date: 2026-03-12

## Summary
Updated evaluation visualization helpers so they only concatenate slots that actually received image tensors, and they fall back to zero confidence panels when confidence maps are absent.

## Motivation
Batch-size-1 evaluation with `num_imgs_vis > 1` left empty sublists inside `get_vis_imgs_new()`, which caused `torch.cat()` to fail after the forward and loss had already succeeded.

## Files Modified
- src/train.py
- changes/README.md

## Notes
This only affects logging and visualization. The training and evaluation losses are unchanged.
