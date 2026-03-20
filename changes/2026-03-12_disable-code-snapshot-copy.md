# Change: Disable training code snapshot copies

Date: 2026-03-12

## Summary
Removed the automatic codebase copy that was created under each checkpoint directory at the start of training.

## Motivation
The repository already tracks changes through git commits and `changes/` logs, so copying the whole worktree into every experiment directory was redundant and cluttered the checkpoint tree.

## Files Modified
- src/train.py
- changes/README.md

## Notes
Checkpoints and logs are still saved normally. Only the extra `code/<timestamp>` snapshot directory is removed.
