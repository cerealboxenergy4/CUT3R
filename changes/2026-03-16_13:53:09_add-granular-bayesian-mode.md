# Change: granular 베이지안 디코더 모드 추가

Date: 2026-03-16_13:53:09

## Summary
`bayes_cut3r`에 `bayesian_mode='granular'`를 추가했다. 이 모드는 decoder layer마다 채널별 alpha를 따로 예측하고, `state_feat`, `current_feat`, `pose_feat`를 함께 사용하는 step-conditioned posterior encoder를 통해 `attn`, `cross_attn`, `mlp`별 alpha를 분리해 적용한다. 또한 granular 모드에서는 alpha history 기반 근사 KL 대신 posterior mean/logvar에 대한 explicit KL을 계산하도록 확장했다.

## Motivation
기존 구현은 pooled state에서 layer별 스칼라 alpha만 예측하는 구조여서 BARNN의 조건부 variational 구조와 차이가 컸다. 별도 모드를 추가해 기존 실험 경로는 보존하면서도, 더 세밀한 posterior conditioning과 명시적 KL regularization을 갖는 실험 경로를 마련할 필요가 있었다.

## Files Modified
src/dust3r/blocks.py
src/dust3r/model.py
scripts/make_bayesian_decoder_init_ckpt.py
config/bayes_kitti_odometry_poc_granular.yaml
changes/README.md

## Notes
- `granular` 모드에서는 `BayesianLinear`가 채널별 alpha를 받아 출력 차원에 맞게 확장할 수 있도록 수정했다.
- decoder block은 동일 alpha를 공유하던 기존 구조와 달리 `attn`, `cross_attn`, `mlp`별 alpha dict를 받을 수 있게 바뀌었다.
- 추론 시 stochastic 샘플링 여부는 기존 `bayesian_sample_inference` 불리언과 호환되도록 유지하되, 새 `bayesian_inference_mode` 설정으로 `mean` 또는 `stochastic`를 명시적으로 선택할 수 있게 했다.
- `torch`가 현재 쉘 환경에 없어 실제 모델 import/instantiate smoke test는 수행하지 못했고, 변경 파일에 대해 `python -m py_compile`만 확인했다.
