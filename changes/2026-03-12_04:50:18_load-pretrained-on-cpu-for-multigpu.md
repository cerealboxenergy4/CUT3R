# Change: 멀티 GPU용 pretrained CPU 로드로 변경

Date: 2026-03-12_04:50:18

## Summary
pretrained checkpoint를 GPU로 직접 읽지 않고 CPU에서 먼저 로드한 뒤 `load_state_dict` 후 모델을 GPU로 옮기도록 학습 코드를 수정했다.

## Motivation
멀티 GPU 실행 시 각 프로세스가 대형 checkpoint를 자기 GPU에 바로 올리며 로드하면 VRAM 사용량이 순간적으로 크게 튀고, 모델 로드 단계에서 멈춘 것처럼 보일 수 있다. CPU 로드 후 GPU 이동 방식이 더 안전하다.

## Files Modified
src/train.py

## Notes
resume checkpoint 경로는 이미 CPU 로드를 사용하고 있었고, 이번 수정은 pretrained 초기화 경로만 동일한 방식으로 맞춘 것이다.
