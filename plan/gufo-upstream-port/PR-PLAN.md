# gufo 원본 PR 분할 계획 (2026-10-25)

상위 문서: [이식 계획](PLAN.md) U6·D2·D3, [U4f 마무리](U4G-TURNKEY.md), [이슈 초안](ISSUE-DRAFT-PAGED-KV.md).

사용자 결정(2026-10-25): "결정 사항 반영해서 진행해줘. 공개(push·PR·이슈)는 아직 하지 마."

## 운영 방식

- **보류(2026-10-25 사용자: "머지말고 일단 분리한상태로 가자")**: 원본에 PR·이슈를 보내지 않는다. 포크 `namote/gufo`(`ai-tools/port-tidy`, `ai-tools/port-u4f`)에서 독자 유지한다. 아래 계획은 나중에 제안할 때 쓴다.

- 우리 브랜치는 계속 독자적으로 유지하고 운영 기준으로 삼는다. 원본에는 독립적인 수정만 올린다(혼합 방식).
- paged KV(③)는 PR보다 이슈로 방향부터 묻는다(U6·D3). 초안: [ISSUE-DRAFT-PAGED-KV.md](ISSUE-DRAFT-PAGED-KV.md). 올리지 않았다.
- 공개(push·PR·이슈)는 D2(공개 계정·이메일)가 정해질 때까지 하지 않는다.

## 정리 브랜치

- gufo `ai-tools/port-tidy` `aeffca08`(33커밋)는 원본 main `82c62b40` 위의 직선형 이력이다. 원래 `ai-tools/port-u4f` `0341dd88`은 그대로 보존했다.
- 원본 main은 정리 시점(2026-10-25)에도 `82c62b40`이다(`gufo-org/gufo` fetch로 확인). 브랜치 전체가 최신 원본 위에 충돌 없이 놓인다.
- 정리 내용:
  - 원본 병합 커밋 `12fbc7ad`를 없애고 직선으로 다시 쌓았다. 병합 충돌 해결은 해당 커밋에 넣었다: CMake·WMMA launch·원본 #705 테스트 호출은 block table 커밋에, `disk.cpp` platform 계층은 warm-up 커밋에 넣었다.
  - `ffc9ca7f`는 뺐다(system 경계 커밋이 대체).
  - system 경계 3커밋(`17163c2f`, `8cd5c24d`, `c0a68ddc`)을 `132bad8e` 하나로 합쳤다.
  - 리뷰 수정 4커밋(`8d50a674`, `923a4194`, `1a017a3f`, `7e3af59e`)은 hunk마다 그 줄을 마지막으로 바꾼 우리 커밋(`git blame`, 수정 커밋끼리는 무시)으로 보냈다. 한 변경이 둘로 갈려 중간 빌드가 깨지지 않게 묶음 단위로 대상을 맞췄다.
    - copy-on-write `first` 규칙(시그니처·본문·테스트) → `50db4581`
    - DrainOnUnwind(정의와 사용처) → paged KV `753e1cc5`
    - 자기 실행의 대기 checkpoint만 기다리기와 WriteBackDeferred, 제출 전 '썼음' 표시 → `3c2f4d77`(U4e 수정, `pending_branch`를 만든 커밋)
    - chunk별 디스크 쓰기·유휴 deadline: 대상(write-back 커밋)에서는 뒤 커밋이 만든 문맥 때문에 충돌해, 깨끗이 적용되는 첫 커밋인 `3c2f4d77`로 미뤘다.
    - 블록 참조 반환·미완료 복사 drain → 블록 보존 `f8c765c7`
    - 미리읽기 재개·문서 → warm-up `4d30a3b1`
    - 미매핑 블록 해제(`7e3af59e`) 전체 → 직접 매핑 `c37affb1`
    - 나머지(커널 서식, 테스트 `TemporaryDirectory`, 문서, 로그)는 각 소유 커밋
  - 메시지는 모두 한 줄 Conventional Commit이고, 작성자는 `Moonki Kang <namote@gmail.com>`이다.
- 남긴 것(사용자 목록 밖이라 합치지 않음): U4e 수정 `3c2f4d77`·`50db4581`, 예전 fix-up `0e382b3f`·`2e8934f7`·`7fb6c1aa`.
- 검증:
  - `35a1b147`(32번째) 트리 = `663d50c9`, HEAD 트리 = `0341dd88`(`git diff` 빈 결과).
  - 새로 빌드한 CPU 테스트 138/138 통과.
  - 정리 도구: `/tmp/claude-1000/…/scratchpad/tidy/`(map.py·build.py·assemble.py, 임시)

## PR 목록

번호는 사용자 지시의 묶음이다. "원본 위 충돌"은 원본 main `82c62b40`(①②는 단독, ③④는 앞 PR을 쌓은 위)에 cherry-pick했을 때의 결과다. "빌드·테스트"는 그 상태에서 CPU 빌드+CTest와 GPU(HIP) 빌드를 돌린 결과다.

|PR|포함 커밋(port-tidy)|원본 위 충돌|빌드·테스트|
|---|---|---|---|
|① ShareCheckpoint 경계 수정 — **제외**(2026-10-25 사용자: "ShareCheckpoint 경계 수정은 우리가 그냥 진행하자", 포크에서 유지)|`5bb34847`|없음(원본 main 단독)|CPU 137/137, GPU 빌드 OK. 새 테스트는 수정 전 코드에서 실패|
|② 관측 로그|`941854a0`(write-back 시간), `6324aa9f`(restore_failed)|없음(원본 main 단독)|CPU 137/137, GPU 빌드 OK|
|③-1 공유 pass scratch|`d8a7b2aa`, `e27b74f1`(마지막 hidden 행만 보존)|없음(원본 main 단독)|CPU 137/137, GPU 빌드 OK|
|③-2 paged KV pool|`aa4bb469`, `0e382b3f`, `2e8934f7`, `753e1cc5`, `000bcd96`|없음|CPU 138/138, GPU 빌드 OK|
|③-3 재배정 capture·유휴 write-back|`28cfeeeb`, `cf11e6c3`, `13657864`, `368c0c3e`, `414f0fbc`, `15700772`, `2512d04d`, `0b825331`, `931167f7`|없음|CPU 138/138, GPU 빌드 OK|
|③-4 디스크 복원 warm-up|`4d30a3b1`|없음|CPU 138/138, GPU 빌드 OK|
|③-5 checkpoint 블록 보존|`f8c765c7`, `7fb6c1aa`, `c02fa4f4`, `2321f5ad`(tail checkpoint)|없음|CPU 138/138, GPU 빌드 OK|
|③-6 공유 prompt 버퍼|`d6346808`|없음|CPU 138/138, GPU 빌드 OK|
|④-1 system prompt checkpoint|`cb00ae91`|없음|CPU 138/138, GPU 빌드 OK|
|③-7 리뷰 수정|`3c2f4d77`, `50db4581`|없음|CPU 138/138, GPU 빌드 OK|
|③-8 직접 매핑·RAM 회계·풀 크기 테스트|`3aabcbf2`, `c37affb1`, `aeffca08`|없음|CPU 138/138, GPU 빌드 OK|
|④-2 system 경계(첫 렌더링 + 끌어올림 probe)|`132bad8e`|없음|CPU 138/138, GPU 빌드 OK|
|④-3 write-back 미리 읽기|`35a1b147`|없음|CPU 138/138, GPU 빌드 OK|

### 사용자 지시 순서와 다른 점 (의존성 때문)

- ③ 안의 순서는 지시와 다르다. 충돌만 확인하는 probe로 여러 순서를 시험했다.
  - **블록 보존(③-5)은 재배정 capture·유휴 write-back(③-3) 뒤에만 올라간다.** 블록 보존 커밋이 write-back·capture 코드를 고친다. 지시 순서(보존 → 재배정)로 쌓으면 `f8c765c7`이 `component_text_cache.cpp`·테스트에서 충돌한다.
  - warm-up(③-4)은 독립적이라 ③-3 뒤에 둔다.
  - U4e 리뷰 수정(③-7)은 system prompt checkpoint(④-1)와 prompt 버퍼(③-6)를 고친다. 그래서 ④-1을 ③-7 앞에 둔다. 직접 매핑(`c37affb1`)은 copy-on-write 수정(`50db4581`) 뒤에만 올라가 ③-8로 뺐다.
  - 이 순서(B5)로 모두 쌓은 트리는 `port-tidy`와 같다.
- B5 순서의 13단계(①·②를 맨 끝에 쌓은 경우 포함) 모두 충돌 없이 올라가고, 단계마다 CPU 빌드+CTest 통과, GPU 빌드 성공이다(`tmp/test-results/gufo-upstream-port-20261023/prcheck/RESULTS-B5b.json`, 단계별 로그).
- 처음 B5 확인에서 ③-3~④-1 단계가 `pending_branch` 없음으로 빌드 실패했다. 리뷰 수정의 한 hunk(자기 실행 대기 checkpoint만 기다리기)가 그 멤버를 만든 커밋보다 앞에 배정된 탓이라, 그 hunk를 `3c2f4d77`로 옮겨 정리 브랜치를 다시 만들었다(트리 동일 재확인).
- 권장 제출 순서:
  1. ①, ②: 원본에도 바로 쓸모 있는 독립 PR. 원본 main 위에서 단독으로 빌드·테스트된다.
  2. ③-1: paged KV와 무관하게 메모리 이득(16K×4 +6.4 GiB, 64K×8 +14.7 GiB). 이슈 답을 기다리지 않고 낼 수 있다.
  3. ③-2 이후: 이슈(D3) 답을 본 뒤 위 순서로.

### 크기

- 가장 큰 커밋은 block table(`aa4bb469`, 35파일 +1,571/−435), paged map(`753e1cc5`, 18파일 +924), 블록 보존(`f8c765c7`, 22파일 +839), system 경계(`132bad8e`, 15파일 +701), U4e 수정(`3c2f4d77`, 18파일 +683)이다. ③-2와 ③-5는 리뷰 부담이 커서 원본 답에 따라 더 쪼갤 수 있다.

## 원본 검증 범위

- 빌드·테스트는 우리 ROCm 7.2.4/GCC 13.3 이미지에서만 했다. 원본 CI(Nix GCC 15.3)는 확인하지 않았다.
- GPU 실행 검증(F16·F17, 단위 9개, 텍스트·T2 동일)은 `port-u4f`(같은 트리)에서 했다. 각 PR 경계에서 GPU 테스트를 돌리지는 않았다.
