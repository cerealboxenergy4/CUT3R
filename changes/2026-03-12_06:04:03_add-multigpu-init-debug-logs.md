# Change: 멀티 GPU 초기화 디버그 로그 추가

Date: 2026-03-12_06:04:03

## Summary
멀티 GPU 학습이 동일한 지점에서 멈추는 문제를 좁히기 위해 `accelerator.prepare()`, resume state 로드, logger 초기화 전후에 명시 로그를 추가했다. 또한 optimizer 파라미터 그룹 로그를 전체 목록 대신 요약 정보만 출력하도록 변경했다.

## Motivation
W&B 단독 초기화는 정상 동작하는 것이 확인되었으므로, 실제 멈춤 지점이 분산 초기화 구간인지 실험 로거 준비 구간인지 구분할 필요가 있었다. 기존 파라미터 그룹 로그는 출력량이 너무 커서 멈춤 위치를 읽기 어렵게 만들고 있었다.

## Files Modified
src/train.py
src/croco/utils/misc.py

## Notes
다음 실행에서 `Starting accelerator.prepare()`, `Finished accelerator.prepare()`, `Logger ready` 중 어디까지 출력되는지 확인하면 NCCL/DDP 초기화 문제인지 여부를 빠르게 판단할 수 있다.
