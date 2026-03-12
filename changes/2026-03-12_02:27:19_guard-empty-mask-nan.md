# Change: 빈 마스크 NaN 방어 및 pose 마스크 정정

Date: 2026-03-12_02:27:19

## Summary
빈 유효 마스크가 들어오는 배치에서 loss detail, 포인트클라우드 정규화, depth 정규화가 `NaN`을 만들지 않도록 방어 코드를 추가했다. 또한 pose loss에 적용되는 `img_mask`가 마지막 view 하나에 의존하던 버그를 전체 view 기준으로 수정했다.

## Motivation
KITTI Bayesian 학습 스모크 런에서 일부 샘플이 비어 있는 valid mask를 만들 수 있고, 이때 실제 최적화 loss는 유한값이어도 로그용 detail과 정규화 통계가 `NaN`으로 오염될 수 있었다. 이 값들이 metric logger와 후속 평가 경로에 전파되면 학습 안정성과 추적성이 떨어진다.

## Files Modified
src/dust3r/losses.py
src/dust3r/utils/geometry.py
src/croco/utils/misc.py

## Notes
문법 검사는 `python -m py_compile`로 확인했다. 전체 smoke train 재실행은 현재 샌드박스에서 CUDA가 비활성화되어 `Half`/RoPE 경로에서 중단되어, 빈 마스크 단위 검증으로 보완했다.
