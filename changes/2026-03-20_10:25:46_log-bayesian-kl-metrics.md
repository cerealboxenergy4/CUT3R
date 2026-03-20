# Change: Bayesian KL 메트릭 로깅 보강

Date: 2026-03-20_10:25:46

## Summary
학습 및 평가 경로에서 베이지안 KL 관련 scalar가 더 일관되게 기록되도록 `src/train.py`를 수정했다. 이제 `bayes_kl`, `bayes_kl_weight`, `bayes_kl_weighted`, `task_loss`를 `kl_weight` 값과 무관하게 `loss_details`에 항상 넣는다.

## Motivation
기존 구현은 `kl_weight <= 0` 분기에서 조기 반환하면서 `bayes_kl` 자체를 `loss_details`에 넣지 않았다. 이 때문에 W&B와 TensorBoard에서 KL 항을 직접 확인하기 어렵고, 콘솔/output log에서도 weighted KL 비중을 추적할 수 없었다.

## Files Modified
src/train.py
changes/README.md

## Notes
- `bayes_kl_weighted`는 실제 total loss에 더해지는 값과 동일한 scalar로 기록된다.
- `metric_logger.update(loss=..., **loss_details)` 경로를 그대로 사용하므로, 주기적 콘솔 로그와 W&B/TensorBoard scalar에 같은 이름으로 노출된다.
- 평가 경로는 여전히 loss 자체에는 KL을 더하지 않지만, `bayes_kl`과 `bayes_kl_weighted=0`은 관측 가능하다.
