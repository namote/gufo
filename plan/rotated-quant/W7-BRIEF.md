# W7 브리프: Flash-Next 확장 준비 조사 (읽기 전용, 2026-10-29)

이 브리프는 [W-COMMON.md](W-COMMON.md)를 따른다. 범위는 HANDOFF의 W7 "준비" 부분과 PLAN의 "R5 이후: Flash-Next 확장 참고" 절이다. 구현은 하지 않는다. 구현은 27B R5 뒤에 사용자 확인을 받고 시작한다.

## 목표

Flash-Next의 dense 트렁크를 Q8_0에서 회전 + 4비트로 내리는 일의 범위, 위험, 비용을 정리한다. 27B 공통 부분(`tools/rotquant/`, `src/core/quant/rotation.*`, `src/core/hip/detail/hadamard.hpp`)을 그대로 쓸 수 있는지, 못 쓰는 곳이 어디인지 밝힌다.

## 조사 항목

1. **로더 지원 타입.**
   - `src/models/qwen38_flash_next/weights.cpp`(SupportedType과 텐서별 허용 목록. dense는 Q8_0, BF16, F16, F32로 보이고, 전문가는 Q5_1, Q8_0 등이다)와 `kernels/rocm/device_model.cpp`, `kernels.hip.cpp`, `mmq/`, `w8a8_wave64.hip`를 읽는다.
   - 텐서 계열마다 실제로 받는 타입과 쓰이는 커널을 표로 만든다.
2. **입력 그룹 표 초안.**
   - 대상: dense attention/SSM 투영, hyper-connection(`hc_*`), gated residual, MTP 헤드, n-gram 테이블, 공유 전문가, 3차원 전문가 텐서(gate/up은 `.rot` 하나를 공유), `down_exps`(K = 640, 1단 포맷만 가능), 출력 헤드(tied 여부)
   - 그룹마다 다음을 적는다.
     - 입력을 만드는 곳(커널 이름과 위치)
     - K
     - 소비 텐서
     - 활성화 정밀도(Q8_1, BF16, FP32, W8A8 등)
3. **활성화 생성 커널 목록.**
   - 경로마다 회전을 어디에 융합해야 하는지 적는다: decode, MTP verify, prefill, graph.
   - 27B와 다른 점을 적는다. 예를 들어 W8A8 경로라면 활성화 양자화 방식이 다르다.
4. **4비트 dense 커널.**
   - Flash-Next에 지금 있는 4비트 GEMV/GEMM이 무엇이고, 무엇이 필요한지 정리한다.
   - #474(gufo-org/gufo)의 Q4_1 dense vec 커널, zengjiajun0623/gufo의 `gufo-opt/flash-next` 브랜치를 검토한다. `gh`로 읽기만 한다. 라이선스와 출처를 기록한다.
   - 27B 커널(`src/models/qwen/hip/kernels/`)을 재사용할 수 있는지 판단한다.
5. **모델 파일과 디스크 계획.**
   - 지원 대상은 `unsloth/Qwen3.8-Flash-Next-GGUF` UD-Q4_K_XL(문서: `docs/models/qwen3.8-flash-next/README.md`)이다. 로컬 `~/projects/ai.tools/storage/models/llm`에도, 외장 백업(`/run/media/m2on/T5 EVO/AI/model/originals-*/llm/`)에도 없다.
   - 사용자 지시(2026-10-29): "외장디스크에 AI 폴더쪽에 모델이 있을꺼야 원본모델.. 그거 써도되고 없으면 받아서 하고". 외장에는 없으므로 내려받는 계획을 세운다.
   - 계획에 넣을 것:
     - 기준선 UD-Q4_K_XL과 MTP 사이드카
     - 회전 변환의 원본. BF16 GGUF가 있는지, 아니면 HF safetensors인지 확인한다.
     - 각 파일의 크기와 리비전
     - 루트 여유 공간 150 GiB 하한 안에서 어떤 순서로 받고 지울지(지금 여유 공간은 약 394 GB)
     - 외장 디스크(exFAT, 여유 약 2 TB)를 원본 보관용으로 쓸 수 있는지
   - **내려받기는 하지 않는다.** 계획만 쓴다. 내려받기는 큰 디스크 쓰기이므로 감독이 조율한다.
6. **#474의 측정 방법.** 에이전트 벤치, 품질 기준, `ssm_out` 민감도를 확인하고, 27B 하네스(W2)를 그대로 쓸 수 있는지 판단한다.

## 파일 경계

- 쓸 수 있는 파일: `plan/rotated-quant/FLASH-NEXT.md`(확장 계획 초안) 하나뿐이다.
- 코드, 빌드, GPU, 내려받기는 하지 않는다.
- 커밋: `docs(plan): …`로 그 파일 하나만 커밋한다.

## 산출물

`FLASH-NEXT.md`에 다음을 담는다.

- 위 1–6의 표
- 공통 코드를 재사용할 때 막히는 곳
- 구현 단계 제안(R2/R4에 대응하는 단계와 통과 기준)
- 사용자 결정이 필요한 것
- 예상 디스크와 GPU 창 비용

짐작과 확인한 것을 구분해서 적는다.
