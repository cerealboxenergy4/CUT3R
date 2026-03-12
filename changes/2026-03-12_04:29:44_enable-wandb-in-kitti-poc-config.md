# Change: KITTI PoC config에 W&B 기본 로깅 활성화

Date: 2026-03-12_04:29:44

## Summary
`bayes_kitti_odometry_poc.yaml`에 W&B 설정을 추가해 별도 커맨드라인 override 없이도 학습 시 자동으로 W&B 로깅이 켜지도록 수정했다.

## Motivation
장시간 학습 실험의 진행 상황과 결과를 커맨드라인이나 로컬 TensorBoard만으로 보기 불편했다. 팀과 프로젝트 이름이 정해진 상태이므로 YAML 기본값으로 활성화하는 편이 실험 추적에 더 적합하다.

## Files Modified
config/bayes_kitti_odometry_poc.yaml

## Notes
W&B entity와 project는 모두 `bayes-cut3r`로 설정했다. 실행 시 별도 `+wandb.enabled=true` override가 더 이상 필요 없다.
