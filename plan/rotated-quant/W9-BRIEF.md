# W9 브리프: 투기 디코딩 개선 (2026-10-29)

이 브리프는 [W-COMMON.md](W-COMMON.md)를 따른다. 근거는 PLAN "방향 전환" 절 2번이다(사용자 지시 2026-10-29). 이 일은 양자화 형식과 무관하므로 W8 측정과 병행한다.

## 배경 (공개 자료, 모두 상대 측정)

- **Flash-Next:** Kyojin(EXL3)은 Strix Halo에서 투기 디코드 52.2/51.1 tok/s(8K/32K)를 냈다. Kyojin이 잰 gufo v0.8.0은 35.3/31.9였다([Kyojin benchmarks.md](https://github.com/Yamz-Labs/kyojin/blob/main/doc/benchmarks.md)).
- **plain decode는 비슷하다:** W7 정리로 Kyojin 27.3, gufo 28.3이다. 차이는 주로 투기 배율에서 온다. Kyojin은 1.7–1.9×이고, gufo Flash-Next는 1.3–1.4×다([FLASH-NEXT.md](FLASH-NEXT.md) 13절).
- **27B:** gufo는 이미 DFlash2로 AR 대비 2.28×다(UD-Q4_K_M, 64문항 텍스트, R0-RESULTS). 그래서 **우선순위는 Flash-Next**이고, 27B는 같은 아이디어가 이득을 주는지만 본다.
- **Kyojin이 쓰는 것(조사 대상):** MTP + n-gram 혼합 드래프트 규칙, 2단 드래프트 헤드, 적응형 드래프트 길이. vcruz305 레시피에는 `ndt`(드래프트 수)와 `dc`(임계값) 같은 손잡이가 있다.

## 1단계: 조사와 설계 (읽기만, 허락 불필요)

1. **gufo 현재 구조:**
   - Flash-Next(`src/models/qwen38_flash_next/`): MTP 헤드, n-gram 테이블, 드래프트 길이 정책, verify 경로, 수락 규칙
   - 27B(`src/models/qwen/`): DFlash2, MTP(nextn), 적응형 정책
   - 각각 표로 만든다: 드래프트 원천, 길이 결정, 트리/체인, verify 배치 크기, 수락 판정
2. **비교 대상 소스(MIT):**
   - Kyojin: `tools/qwen/serve.py`, 투기 관련 모듈
   - upstream exllamav3: speculative, MTP
   - vcruz305 `exllamav3-amd`: MTP 손잡이
   - 정리할 것: 혼합 규칙(언제 n-gram, 언제 MTP), 2단 헤드의 구조와 학습 여부(가중치를 더 쓰는지), 적응형 길이의 신호, 카테고리별(prose/chat/code) 배율
   - 라이선스와 출처를 기록한다.
3. **gufo 기준 측정 설계:**
   - W2의 `rq_serving.py`와 `accept_prompts.json`(64문항)을 쓰고, `accept.py`로 프롬프트 단위 paired 구간을 낸다.
   - Flash-Next는 지금 UD-Q4_K_XL 파일이 외장 디스크에 있다(`/run/media/m2on/T5 EVO/AI/model/originals-flash-next-20261029/`). 로컬 복사 여부와 디스크(루트 150 GiB 하한)를 계획에 넣는다.
   - 지표: 출력 토큰당 수락 토큰, verify 스텝당 수락, 실효 tok/s, 카테고리별 값, 문맥 깊이(8K, 32K)
4. **개선안과 예상 이득:** 항목마다 예상 배율, 구현 크기, 위험, 품질 영향(greedy 출력이 같아야 함)을 적는다. 순서를 제안한다.
5. `plan/rotated-quant/SPEC-DECODE.md`에 쓰고 `WAITING: 구현 승인`으로 보고한다.

## 2단계: 구현 (감독 승인 뒤)

- **출력 동일성:** 투기 디코딩은 greedy 출력을 바꾸면 안 된다. 같은 프롬프트에서 AR과 투기 출력이 같은지 확인하는 기존 기능 테스트를 찾아 돌린다(`tests/functional/`, `long-context`, `cache` 포함).
- **판정:** 속도는 CLAUDE.md 규칙(요청·단계별 5 %·10 ms, main과 짝지은 대조, ABBA)을 따른다.
- **파일 충돌:** `src/models/qwen/`는 W3가 R2 로더 일을 마무리하는 중이다. 27B 쪽을 고치기 전에 감독에게 순서를 확인한다. Flash-Next 쪽(`src/models/qwen38_flash_next/`)은 W9만 고친다.

## 파일 경계

- 1단계에서 쓸 수 있는 파일은 `plan/rotated-quant/SPEC-DECODE.md` 하나다.
- 2단계 파일 경계는 감독이 승인할 때 정한다.
- 빌드, GPU, 모델 복사는 감독 허락 뒤에만 한다.

짐작과 확인한 것을 구분해서 적는다. 근거에는 코드 위치(파일:줄)와 URL을 단다.
