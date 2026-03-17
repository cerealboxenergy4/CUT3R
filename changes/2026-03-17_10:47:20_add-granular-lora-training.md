# Change: granular 베이지안 디코더에 LoRA 학습 옵션 추가

Date: 2026-03-17_10:47:20

## Summary
`bayes_cut3r`에 decoder/state decoder용 LoRA 학습 옵션을 추가했다. `BayesianLinear`가 low-rank adapter를 가질 수 있도록 확장했고, 모델 config에서 LoRA rank/alpha/dropout과 적용 범위를 제어할 수 있게 했다. 또한 posterior alpha encoder와 LoRA만 함께 학습하는 `dropout_encoder_lora_only` freeze 모드를 추가하고, 실행용 `granular + LoRA` config를 새로 만들었다.

## Motivation
기존 `granular` 설정은 posterior alpha encoder만 학습하고 원래 decoder/state decoder weight는 모두 고정되어 있었다. 원본 모델 weight를 직접 풀지 않으면서도 표현력을 조금 더 조정할 수 있도록, low-rank adapter 방식의 가벼운 fine-tuning 경로가 필요했다.

## Files Modified
src/dust3r/blocks.py
src/dust3r/model.py
config/bayes_kitti_odometry_poc_granular_lora.yaml
changes/README.md

## Notes
- LoRA는 decoder block과 state decoder block 내부의 `BayesianLinear`에만 붙도록 구현했다.
- 기본 실행 config는 `freeze='dropout_encoder_lora_only'`, `use_lora=True`, `lora_rank=4`, `lora_alpha=16.0`, `lora_dropout=0.0`를 사용한다.
- 기존 granular init checkpoint를 그대로 사용할 수 있고, LoRA 파라미터는 `strict=False` 로드에서 새로 초기화된다.
- smoke test 기준으로 작은 모델에서 LoRA가 붙은 `BayesianLinear` 32개가 확인되었고 trainable parameter 수는 651,687개였다.
