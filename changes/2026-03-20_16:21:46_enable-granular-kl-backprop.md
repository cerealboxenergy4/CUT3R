# Change: granular KL 역전파 복원

Date: 2026-03-20_16:21:46

## Summary
`src/dust3r/model.py`의 granular 베이지안 통계 누적 경로에서 `kl_loss.detach()`를 제거했다. 이제 granular 모드의 KL 합/평균이 live tensor로 유지되어 total loss에 더해질 때 alpha encoder와 posterior 경로로 실제 gradient가 전달된다.

## Motivation
기존 구현은 granular KL을 통계에 누적할 때 detach해서 저장하고 있었고, 그 결과 로그에는 `bayes_kl_weighted`가 커도 backward에서는 KL regularization이 전혀 작동하지 않았다. variational regularizer의 의도대로 KL 항이 posterior encoder를 실제로 제약해야 한다.

## Files Modified
src/dust3r/model.py
changes/README.md

## Notes
- alpha mean/var/min/max 통계는 여전히 detached 경로라 logging 용도로만 유지된다.
- KL 텐서는 현재 forward 범위 안에서만 live graph를 유지하므로, granular 모드의 VRAM 사용량은 일부 증가할 수 있다.
- 다음 train 로그에서는 `bayes_kl_weighted`가 0이 아니면서, 실제 backward에도 반영된다.
