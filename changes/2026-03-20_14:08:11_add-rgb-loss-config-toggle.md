# Change: RGB loss config 토글 추가

Date: 2026-03-20_14:08:11

## Summary
`src/train.py`에서 train/test criterion 문자열을 구성할 때 `RGBLoss(...)` 항을 config 기반으로 제거할 수 있도록 수정했다. 기본값은 `use_rgb_loss: false`이며, 필요하면 `train_use_rgb_loss`, `test_use_rgb_loss`로 split별 제어도 가능하다.

## Motivation
현재 실험에서는 train과 test 모두 RGB loss를 사용하지 않길 원하지만, criterion 문자열을 매번 직접 수정하는 방식은 번거롭고 추적이 어렵다. config에서 토글할 수 있어야 같은 config를 유지한 채 loss 구성을 바꿀 수 있다.

## Files Modified
src/train.py
changes/README.md

## Notes
- 시작 로그에는 실제 적용된 criterion 문자열이 출력되므로 `RGBLoss(...)` 제거 여부를 바로 확인할 수 있다.
- `use_rgb_loss: true`로 두 split 모두 다시 켤 수 있다.
- `train_use_rgb_loss`, `test_use_rgb_loss`를 따로 주면 split별로 다르게 설정할 수 있다.
