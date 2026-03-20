# Change: 모델에 베이지안 KL config 복원

Date: 2026-03-20_15:31:46

## Summary
`src/dust3r/model.py`에서 `ARCroco3DStereo` 초기화 후 custom config가 `CroCoNet` 상위 초기화에 덮이지 않도록 수정했다. 또한 `bayesian_kl_weight`를 모델 본체 속성으로도 저장해 train/eval 로깅과 loss 적용 시 안정적으로 읽을 수 있게 했다.

## Motivation
학습 로그에서 `bayes_kl_weight`와 `bayes_kl_weighted`가 train에서도 0으로 찍혔고, 실제 total loss에도 KL 항이 더해지지 않았다. 원인은 `ARCroco3DStereoConfig`의 `bayesian_kl_weight`가 상위 클래스 초기화 이후 `model.config`에서 사라져 `train.py`가 0 fallback을 사용하던 구조였다.

## Files Modified
src/dust3r/model.py
changes/README.md

## Notes
- 이제 `model.config`는 다시 `ARCroco3DStereoConfig`를 가리킨다.
- `model.bayesian_kl_weight`도 직접 보존하므로 helper가 config overwrite 상황에서도 안전하게 읽을 수 있다.
- 다음 run에서는 train 로그에서 `bayes_kl_weighted`가 0이 아니어야 정상이다.
