# Change: granular 모드 VRAM 사용량 절감

Date: 2026-03-16_14:21:25

## Summary
`granular` 베이지안 모드에서 VRAM을 크게 쓰던 두 경로를 줄였다. 첫째, decoder block에 alpha를 dict 대신 텐서 형태로 넘겨서 gradient checkpointing이 다시 활성화되도록 수정했다. 둘째, step별 alpha history 전체를 저장하던 방식을 없애고, detached running statistics와 KL 누적합만 유지하도록 바꿨다.

## Motivation
같은 `num_views` 설정에서도 granular 모드가 기존보다 훨씬 많은 VRAM을 사용했다. 원인은 checkpoint 비활성화와 view-step마다 `[B, dec_depth, 3, dec_embed_dim]` alpha 텐서를 모두 보관하던 구조였다. 메모리 병목을 줄이기 위해 두 부분을 직접 수정할 필요가 있었다.

## Files Modified
src/dust3r/blocks.py
src/dust3r/model.py
changes/README.md

## Notes
- `granular` alpha는 이제 `[B, dec_depth, 3, D]` 텐서로 유지되고, 블록 내부에서 `attn/cross_attn/mlp` 인덱스로 선택한다.
- 이 변경으로 decoder checkpoint 경로가 다시 사용 가능해졌다.
- 통계 계산은 `count/sum/sum_sq/min/max`와 KL 평균 누적 방식으로 바뀌었고, 전체 alpha history stack은 granular 모드에서 더 이상 쌓지 않는다.
- 파라미터 shape는 바뀌지 않아 granular init 체크포인트는 다시 만들 필요가 없다.
