# Change: state-recur smoke test용 배치/저장 버그 수정

Date: 2026-03-18_10:55:09

## Summary
`state-recur` 스모크 테스트 중 드러난 두 가지 기존 버그를 수정했다. 첫째, `batch_size=1`일 때 pose loss 계산에서 `pose_masks`가 scalar로 붕괴되던 문제를 고쳤다. 둘째, `save_freq=0`을 disable 값으로 사용하면 train loop에서 modulo-by-zero가 발생하던 문제를 막았다.

## Motivation
sequence state recurrence 자체는 동작했지만, 작은 스모크 학습을 돌리는 과정에서 unrelated bug가 먼저 학습을 멈췄다. `state-recur` 경로를 검증하고 이후 실험을 안정적으로 돌리려면 이 두 방해 요소를 먼저 제거할 필요가 있었다.

## Files Modified
src/dust3r/losses.py
src/train.py
changes/README.md

## Notes
- `pose_masks`는 `squeeze()` 대신 `reshape(-1)`를 사용해 batch 차원을 유지하도록 바꿨다.
- `save_freq=0`일 때는 step-level intermediate save를 완전히 건너뛰도록 guard를 추가했다.
- 수정 후 `granular + LoRA + state_recur` 조합으로 KITTI 소형 subset smoke train을 `batch_size=1`과 `batch_size=2`에서 모두 끝까지 통과시켰다.
