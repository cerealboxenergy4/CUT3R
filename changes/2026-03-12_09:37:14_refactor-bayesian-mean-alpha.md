# Change: Bayesian mean alpha formulation으로 재구성

Date: 2026-03-12_09:37:14

## Summary
Bayesian decoder를 `alpha`가 mean 경로에도 직접 반영되는 형태로 재구성했다. `BayesianLinear`는 이제 inference에서 `alpha * (xW) + b`를 사용하고, training에서만 고정 `sigma=1e-3` 스케일의 noise를 추가한다. 또한 `dropout_encoder_only` freeze 모드와 bayesian dropout encoder 재초기화 경로를 추가했고, 관련 PoC config와 init checkpoint 메타데이터를 새 의미에 맞게 갱신했다.

## Motivation
기존 구현은 training에서만 분산 항에 `alpha`가 들어가고 inference mean은 항상 원래 weight를 사용했다. 이 구조에서는 `dropout_encoder`만 학습해도 inference 출력이 바닐라 모델과 같아서, alpha 학습 결과가 추론에 반영되지 않았다. mean 경로에도 alpha를 반영하도록 바꾸고, 초기 alpha를 1 근처로 맞춰 pretrained 동작을 최대한 보존하는 형태가 필요했다.

## Files Modified
- `src/dust3r/blocks.py`
- `src/dust3r/model.py`
- `src/train.py`
- `config/bayes_kitti_odometry_poc.yaml`
- `config/bayes_kitti_odometry_poc_dropout_encoder_only.yaml`
- `changes/2026-03-12_09:37:14_refactor-bayesian-mean-alpha.md`
- `changes/README.md`

## Notes
- `BayesianLinear`의 deterministic 경로는 `alpha=1`일 때 기존 linear와 정확히 일치하도록 확인했다.
- `dropout_encoder`의 마지막 linear는 새 semantics에 맞게 `weight=0`, `bias=softplus^{-1}(alpha_init - alpha_min)`로 재초기화되도록 변경했다.
- 학습 시작 시 `reset_bayesian_dropout_encoder=true`를 사용하면 pretrained checkpoint에 들어 있던 구식 `dropout_encoder` 파라미터를 무시하고 새 의미로 다시 초기화한다.
- 저장소 밖의 외부 체크포인트 `/home/hunn/projects/checkpoints/cut3r_512_dpt_4_64_bayes_decoder_init.pth` 도 동일한 새 semantics에 맞게 덮어써서, 이후 실험이 같은 init state에서 시작하도록 맞췄다.
