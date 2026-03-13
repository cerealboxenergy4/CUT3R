# Change: 학습 autocast를 bf16으로 전환

Date: 2026-03-13_10:08:50

## Summary
`use_amp=True`일 때 DUSt3R/CUT3R 학습 forward 경로가 기본 fp16 autocast 대신 bf16 autocast를 사용하도록 변경했다.

## Motivation
`poc_test_DE_00` 재현에서 `dropout_encoder`가 step 1 backward 이후 all-NaN gradient로 무너지는 문제가 있었고, 동일 조건의 짧은 재현에서 bf16 autocast 경로는 같은 시점까지 gradient가 finite로 유지되는 것을 확인했다. 기존 구현은 `torch.cuda.amp.autocast(enabled=...)`를 사용해 사실상 fp16 경로를 탔기 때문에, 숫자 범위가 더 넓은 bf16으로 명시적으로 전환할 필요가 있었다.

## Files Modified
- src/dust3r/inference.py
- changes/2026-03-13_10:08:50_switch-training-autocast-to-bf16.md
- changes/README.md

## Notes
- `loss_of_one_batch()`와 `loss_of_one_batch_tbptt()`의 forward autocast를 `torch.amp.autocast("cuda", dtype=torch.bfloat16, ...)`로 바꿨다.
- criterion/loss 계산은 기존처럼 autocast 비활성화 상태를 유지했다.
- 수정 후 `python -m py_compile src/dust3r/inference.py`를 통과했다.
- 동일한 `DE_00` 설정으로 2-step 재현을 수행했고, 두 step 모두 `dropout_encoder` gradient가 finite로 유지되는 것을 확인했다.
