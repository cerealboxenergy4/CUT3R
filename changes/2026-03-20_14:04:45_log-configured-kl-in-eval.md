# Change: Eval에서 configured KL 지표 기록

Date: 2026-03-20_14:04:45

## Summary
`src/train.py`에 베이지안 KL weight 조회 helper를 추가하고, 학습/평가 로깅에 configured KL 지표를 함께 남기도록 수정했다. train에서는 configured 값과 실제 적용 값이 동일하게 기록되고, test에서는 실제 적용 값은 0으로 유지하되 `bayes_kl_weight_configured`, `bayes_kl_weighted_configured`로 설정 기준 KL 크기를 확인할 수 있다.

## Motivation
SLURM 콘솔 로그에서 `bayes_kl_weight`가 소수점 4자리 포맷 때문에 0처럼 보이거나, evaluation에서는 weighted KL이 항상 0으로 찍혀 해석이 어려웠다. test에서도 raw KL과 configured-weighted KL을 같이 확인할 수 있어야 실험 추적이 수월하다.

## Files Modified
src/train.py
changes/README.md

## Notes
- `bayes_kl_weight`와 `bayes_kl_weighted`는 실제 loss에 적용된 값을 계속 의미한다.
- 새 `bayes_kl_weight_configured`와 `bayes_kl_weighted_configured`는 모델 config 기준 값이라 eval에서도 0이 아니다.
- train에서 configured 값과 applied 값이 다르면 weight 조회 경로 문제를 바로 식별할 수 있다.
