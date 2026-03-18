# Change: 시퀀스별 state recurrence cache 추가

Date: 2026-03-18_10:31:35

## Summary
학습 중 같은 시퀀스에서 나온 마지막 `state_feat`를 cache에 저장하고, 다음 배치 시작 시 확률적으로 초기 state로 재사용하는 `state_recur` 경로를 추가했다. reset 동작은 기존 fresh init을 유지하고, cached state는 시작점에서만 주입되도록 구현했다. cache는 gradient graph를 끊기 위해 항상 `detach()` 후 CPU FP16으로 저장한다.

## Motivation
granular + LoRA 설정에서 모델이 가능한 한 다양한 state 분포를 보도록 만들고 싶었지만, 매 배치마다 첫 프레임으로만 state를 초기화하면 같은 시퀀스 안에서 형성되는 state 다양성을 충분히 활용하기 어렵다. 반면 전역 마지막 state를 그대로 넘기는 방식은 시퀀스 간 오염 위험이 커서, 같은 sequence key에 한정된 recurrence cache가 더 안전하다.

## Files Modified
src/dust3r/model.py
src/dust3r/inference.py
src/train.py
config/bayes_kitti_odometry_poc_granular_lora_state_recur.yaml
changes/README.md

## Notes
- `state_init_override` / `state_init_mask`를 모델 forward와 inference helper에 추가했다.
- cached init은 batch 시작 current state에만 적용되고, `reset` 시 복귀점은 fresh first-frame init을 유지한다.
- 현재 KITTI odometry training dataset은 `reset=False`를 사용하므로, 이 설계가 현재 실험 설정과 잘 맞는다.
- cache key는 `dataset::label` 형식을 사용한다.
- state cache는 process-local이다. DDP에서 rank 간 state를 공유하지 않는다.
