# Change: granular posterior 입력을 concat-only로 변경

Date: 2026-03-16_14:11:20

## Summary
`granular` 모드의 posterior encoder 입력 구성을 `state_feat + current_feat + pose_feat` concat-only 형태로 변경했다. 기존에는 `state/current/pose/delta`를 모두 decoder 차원 기준으로 맞춘다고 가정했지만, 실제 학습 경로에서 `current_feat`가 encoder 차원(`1024`)이라 shape mismatch가 발생했다.

## Motivation
실행 중 `pooled_current - pooled_state`에서 `1024`와 `768` 차원 불일치로 학습이 중단됐다. 별도 projection 모듈을 추가하면 추가 파라미터와 freeze 범위 해석이 복잡해지므로, 우선은 projection 없이 concat-only 입력으로 posterior encoder를 구성하는 쪽이 더 단순하고 안정적이다.

## Files Modified
src/dust3r/model.py
changes/README.md

## Notes
- `granular` posterior encoder의 입력 차원은 `dec_embed_dim + enc_embed_dim + dec_embed_dim`으로 바뀌었다.
- 새 입력 shape에 맞춰 `cut3r_512_dpt_4_64_bayes_decoder_init_granular.pth`도 같은 이름으로 다시 생성했다.
- 체크포인트 재생성 자체는 저장소 커밋에 포함하지 않았다.
