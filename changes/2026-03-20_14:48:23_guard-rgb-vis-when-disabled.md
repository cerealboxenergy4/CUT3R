# Change: RGB 비활성 시 시각화 경로 보호

Date: 2026-03-20_14:48:23

## Summary
`src/train.py`의 `get_vis_imgs_new()`에서 `pred_rgb_*` 또는 `gt_img_*`가 없을 때도 evaluation/train 시각화가 실패하지 않도록 수정했다. RGB 관련 키가 없으면 depth map 해상도에 맞는 빈 RGB 패널을 생성해 기존 레이아웃을 유지한다.

## Motivation
RGB loss를 config로 끈 뒤 evaluation 단계에서 `pred_rgb_1` 키가 없어서 `KeyError`가 발생했다. RGB supervision을 비활성화한 실험에서도 depth/pose 지표와 시각화는 계속 확인할 수 있어야 한다.

## Files Modified
src/train.py
changes/README.md

## Notes
- 이제 RGB loss를 끈 실험에서도 `print_img_freq` 또는 eval 시각화 경로가 그대로 동작한다.
- RGB 패널은 빈 이미지로 채워지므로, depth/pose 시각화 레이아웃은 유지되지만 RGB 예측 자체는 표시되지 않는다.
