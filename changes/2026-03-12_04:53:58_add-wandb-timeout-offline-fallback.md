# Change: W&B 초기화 timeout 및 offline fallback 추가

Date: 2026-03-12_04:53:58

## Summary
W&B online 초기화가 지연되거나 실패할 때 일정 시간 후 offline 모드로 자동 전환되도록 학습 코드와 KITTI PoC config를 수정했다.

## Motivation
멀티 GPU 학습에서 모델 로드 직후 `wandb.init()` 단계가 네트워크 상태에 따라 오래 대기하면서 학습이 멈춘 것처럼 보일 수 있었다. 실험 시작 자체가 막히지 않도록 timeout과 fallback이 필요했다.

## Files Modified
src/train.py
config/bayes_kitti_odometry_poc.yaml

## Notes
기본값은 `init_timeout: 30`, `allow_offline_fallback: true`다. online 연결이 실패하면 경고를 출력한 뒤 offline run으로 계속 진행한다.
