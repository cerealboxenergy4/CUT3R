# Change: per-view pose supervision 복구

Date: 2026-03-12_04:12:54

## Summary
pose loss 마스크를 batch 단위 `all(img_mask)`에서 view 단위 `img_mask`로 변경했다. 이제 64-view 설정에서도 실제 이미지가 있는 view들에 대해 pose supervision이 계속 들어간다.

## Motivation
64-view KITTI 설정에서는 각 view가 `img_mask=False`가 될 확률이 누적되어, 모든 view가 동시에 `True`인 샘플이 거의 사라진다. 그 결과 pose loss가 사실상 0이 되어 inference 시 pose가 identity로 콜랩스할 수 있었다.

## Files Modified
src/dust3r/losses.py

## Notes
`compute_pose_loss`가 `BxN` 마스크를 처리하도록 확장했다. 간단한 단위 검증으로 view 일부만 활성화된 마스크에서도 pose loss가 정상 계산되는 것을 확인했다.
