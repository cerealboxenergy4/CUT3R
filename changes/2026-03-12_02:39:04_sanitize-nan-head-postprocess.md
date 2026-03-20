# Change: head postprocess NaN 출력 방어

Date: 2026-03-12_02:39:04

## Summary
prediction head의 후처리 단계에서 `NaN`/`Inf` 값을 안전한 기본값으로 치환하도록 수정했다. depth, confidence, RGB, pose, descriptor 모두 finite 값만 loss로 전달되도록 보강했다.

## Motivation
학습 중 일부 배치에서 head 출력이 이미 `NaN`이 되어 `conf_self`, `conf`, `rgb`, `camera_pose`가 그대로 loss에 들어가며 학습이 중단됐다. 근본 원인은 추가 추적이 필요하지만, 우선 후처리 단계에서 비정상값 전파를 차단해 스모크 런이 즉시 중단되지 않도록 할 필요가 있었다.

## Files Modified
src/dust3r/heads/postprocess.py

## Notes
`python -m py_compile src/dust3r/heads/postprocess.py`를 통과했고, `NaN` 입력 텐서를 넣는 단위 검증에서 depth/conf/rgb/pose 출력이 모두 finite임을 확인했다.
