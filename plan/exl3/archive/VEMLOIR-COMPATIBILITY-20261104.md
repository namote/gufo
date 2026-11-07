# Vemloir Flash-Next EXL3 팩 호환성 문제

> 이 문서는 2026-11-04 정적 조사 당시의 기록이다. 이후 팩 호환성 수정과 c03 변환·검증이 진행됐으며, 현재 상태는 [최종 보정·검증 기록](../F3-MTP-CALIBRATION.md)을 따른다. 아래 코드·서버 상태 설명은 당시 기준이다.

확인일: 2026-11-04. 코드와 모델 메타데이터를 정적으로 확인한 결과다.
이 점검에서는 모델 로딩·추론·GPU 테스트를 실행하지 않았다.

## 대상과 현재 설정

- 모델: `Vemloir/Qwen3.8-Flash-Next-sc-exl3`, `4.05bpw_sc_h8_ng8`
- 리비전: `fe8ad64bebc6de4ae9cb1eee4fd37342c0a51bc7`
- 내부 경로: `/home/m2on/projects/ai.tools/storage/models/llm/flash-next-exl3-vemloir-4.05-sc-h8-ng8`
- Gufo 실행 스크립트: `/home/m2on/.local/share/gufo/gufo-server.py`
- 기본 모델 경로는 위 팩으로 변경했다. 서버는 정지 상태로 유지했다.
- 선택 이미지: `ai-tools/gufo-eval:native-exl3-vision-rocm101`
- 설정: C2, 세션별 native 256K, FP16 공유 KV 풀, YaRN 꺼짐, 내장 비전, MTP, medium.

**현재 구현으로는 이 팩을 그대로 로딩할 수 없다.** 모델 경로 변경과 팩의 실행 호환성 확보는 별개다.

## 확인된 차단 요인

### 1. 분할 ngram 파일을 읽지 못함

팩에는 다음 두 파일이 있으며, `ngram_embedding.safetensors`는 없다.

- `ngram_embedding-00001-of-00002.safetensors`
- `ngram_embedding-00002-of-00002.safetensors`

두 파일의 헤더는 `format=exl3_ngram_trellis`, `version=1`, `K=8`,
`codebook=mul1`, `row_dim=160`이다.

현재 `src/models/qwen38_flash_next/native_metadata.cpp`의 `OpenArtifactMetadata`
경로는 단일 파일명을 필수 입력으로 요구하고 그 파일만 연다.
`src/models/qwen38_flash_next/ngram.cpp`의 호출 경로도 같은 파일명을 사용한다.
`exl3_ngram.cpp`의 `Exl3Ngram::Open`은 한 파일 내부의 shard 텐서를 처리하지만,
여러 safetensors 파일에 걸친 테이블을 연결하는 구현은 아니다.

이 문제가 시작 단계의 첫 번째 차단 요인이다. 단순 파일명 변경으로 해결되지 않는다.

### 2. 언어 모델의 K8 투영을 거부함

새 팩의 `lm_head`는 8비트이며, 언어 모델 내부에도 K8 투영이 있다.

`src/models/qwen38_flash_next/exl3_weights.cpp`의 `BindExl3LinearBank`는
정수 K3~K7만 허용한다. `Binder::Linear`를 통해 출력층과 일반 투영도
이 검사를 거치므로 K8 출력층 로딩이 거부된다.

관련 GPU dispatch와 연산 경로도 확인하고 K8을 지원해야 한다.
로더의 상한 검사만 제거해서는 안 된다.

### 3. K3.5 전문가를 거부함

팩의 `quantization_config.json`에는 3.5비트 언어 모델 투영 8,192개가 있다.
예를 들어 `model.language_model.layers.1.mlp.experts.0.gate_proj`는
`bits_per_weight=3.5`, `codebook=mul1`이다.

- `BindExl3LinearBank`는 `lin.rate.half`를 거부한다.
- `src/models/qwen38_flash_next/kernels/rocm/exl3_moe.hip`의 `WithRate`도
  반정수 비트를 거부하고 정수 K3~K7만 dispatch한다.
- 공통 EXL3 포맷 코드에는 mul1 반정수 비트 처리가 있지만,
  Flash-Next 전문가 실행 경로까지 연결되어 있다는 뜻은 아니다.

MoE의 decode·prefill 경로에서 K3.5 packing, stride, dispatch를 함께 지원해야 한다.
같은 expert bank 안의 비트율 일관성 검사도 유지해야 한다.

## 이미 확인한 부분과 남은 불확실성

- 모델 인덱스가 가리키는 파일들은 존재한다. 내부 복사 때 28개 파일의 해시 검증을 완료했다.
- 내장 K8 비전은 이전 작업에서 독립 참조와 수치 검증을 완료했다.
  이것은 언어 모델 전체·MTP 실행 검증이 아니다.
- MTP mixer의 `hc_norm`, `input_mix_weight_down`, `input_mix_weight_up` 텐서는
  인덱스에 존재한다. MTP 전체의 형식·수치·실행 호환성은 아직 미검증이다.
- 현 설정에서의 전체 로딩 메모리, 긴 컨텍스트 안정성, 속도는 아직 측정하지 않았다.

## 수정 및 검증 순서

1. 분할 ngram 파일을 복사·병합하지 않고 읽도록 파일 간 shard 연결과
   메타데이터·행 범위 검증을 구현한다. 기존 단일 파일도 지원한다.
2. Flash-Next K8 출력층·일반 투영의 로더와 실행 경로를 지원한다.
3. K3.5 전문가의 로더·GPU decode·prefill 경로를 지원한다.
4. 작은 CPU/연산 검사로 형식, 수치, 누락·중복 shard, 경계 조건을 확인한다.
5. 변경을 모은 뒤 이미지에 반영하고, 전체 로딩·정확성·prefill/TG/MTP 검증을
   한 번에 진행한다. 기존 baseline과 Nix 10.1 증분 산출물을 재사용한다.

사용자의 현재 지시는 실행 없이 기록하는 것이다. 이 문서 작성으로
서버 시작이나 GPU 검증을 수행하지 않는다.
