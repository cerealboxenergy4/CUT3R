# Change: 선택적 Weights & Biases 로깅 추가

Date: 2026-03-12_04:24:57

## Summary
학습 코드에 TensorBoard와 병렬로 사용할 수 있는 선택적 W&B 로깅을 추가했다. scalar metric은 step 기준으로 기록되고, 기존 시각화 이미지도 W&B run에 함께 업로드할 수 있다.

## Motivation
현재 학습 정보는 로컬 TensorBoard와 `log.txt` 중심이라 장시간 실험 비교와 원격 모니터링이 불편하다. 기존 logging 지점을 재사용해 W&B를 opt-in 방식으로 붙이면 실험 추적성이 좋아진다.

## Files Modified
src/train.py

## Notes
현재 `bayes_cut3r` 환경에는 `wandb` 패키지가 설치되어 있지 않다. 사용 시에는 먼저 패키지를 설치한 뒤 Hydra override 예시처럼 `+wandb.enabled=true +wandb.project=bayes_cut3r` 형태로 실행하면 된다.
