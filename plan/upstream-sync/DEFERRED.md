# 미뤄 둔 일 계획 (2026-11-09, U6 뒤)

순서: CLEANUP 남은 항목(B1 KVarN, C4, C8a 동시 실행 수 제한) 구현과 리뷰 →
아래 GPU 패스 한 번 → 결과를 보고 고친다. 느려진 것 찾기를 먼저 한다.
운영 서버는 사용자가 말할 때까지 꺼 둔다.

## 1. 느려진 것 찾기 (먼저)

| # | 무엇 | 지금 아는 것 | 확인 방법 |
|---|---|---|---|
| S1 | MTP가 11/08 qualification보다 약 10% 낮다 | base `166953f0`(같은 binary, 같은 작업과 hash): 11/08 65.7/104.4, U6 drop+compact 뒤 58.9–59.3/94–99 tok/s(d0/d32K). 단편화로 39/68까지 떨어졌다가 대부분 돌아왔다. 남은 10%의 원인은 모른다 | (1) `compaction_proactiveness=50`을 적용한 지금 상태로 base MTP d0/d32K를 drop+compact 뒤 3번 잰다. (2) 재부팅 뒤 같은 측정을 한다(**재부팅은 사용자 승인이 필요하다**). 재부팅해서 65대로 돌아오면 환경 문제다(단편화, THP, page cache). 그러면 hugepage를 미리 잡을지 정한다. 돌아오지 않으면 11/08 이후 바뀐 것(firmware, kernel 모듈, docker image, HC session의 pack 변경)을 하나씩 대조한다. /proc/buddyinfo, THP 상태(`/sys/kernel/mm/transparent_hugepage`), GPU clock을 같이 기록한다 |
| S2 | batch pair 행 f64690(`batch_c2_greedy_short_pairs_match_serial`)이 cand4에서 wall 232→379 ms, decode 92→239 ms | cand4 한 번만 나왔다. cand2는 223.7 ms, fb→fc2도 같았다. residents 게이트(`6e2c504e`)는 두 요청이 같이 있으면 wide를 끄지만, base에는 wide가 없어 이것만으로는 base보다 느린 이유가 안 된다 | cand2와 cand4를 번갈아 5번씩 `batch`를 돌린다(drop+compact 뒤). 재현되면 그 행만 `GUFO_PROFILE`/serve 로그로 pass별 시간을 보고, residents 수가 바뀌는 순간(한쪽이 끝날 때) graph 재녹화나 scratch 할당이 생기는지 본다 |
| S3 | cand4 batch replies 행 +5~+25 ms(57개 가운데 14개 flag) | 합계는 base보다 빠르다(6,932 대 7,269 ms). 행별 5%/10 ms 기준에는 걸린다 | S2와 같은 반복에서 같이 본다. 반복해도 같은 행이 걸리면 S2처럼 pass별로 본다. 아니면 판단 보류로 기록한다 |
| S4 | `*-disk` 복원 prefill +10~15 ms(B3) | U5 후보부터 있고 M1 탓은 아니다. CLEANUP B3는 정렬 tail 때문에 kv_pool에서 pass가 하나 더 나뉘는 것으로 본다. U6 보고는 "원인 모름"이다 | 해당 행의 serve 로그에서 prefill pass 수와 경계를 base와 비교해 원인부터 확정한다. 확정되면 B3의 2번(짧은 정렬 tail은 capture하지 않음) 후보를 만들어 `pi_agent.py`(tool loop)와 일반 chat(thinking을 지운 history) 두 workload로 잰다. 둘 다 나빠지지 않을 때만 채택한다 |
| S5 | C4: 다른 세션의 긴 prefill 동안 capture가 7~10 ms에서 1.83 s로 늘어난다 | CLEANUP 구현이 진행 중이다 | U6과 같은 `--sessions 2` 시나리오로 capture 시간과 TTFT를 잰다 |

## 2. 그 밖에 미룬 것 (느려진 것 다음)

| # | 무엇 | 메모 |
|---|---|---|
| D1 | U4-3 E1~E4 replay | hang 재현 도구. 우회가 통하므로 급하지 않다. `u6-rocgdb-nix` docker volume을 남겨 뒀다 |
| D2 | AMD에 MES queue 정지 버그 보고 초안 | 외부 공개라 사용자가 원할 때만 한다. 증거: `build/b1/u6-20261109/hang/` |
| D3 | `mtp_audit` base 대조 | 모든 후보에서 attention-state 0.0071(기준 0.002)로 실패한다. base에서 한 번 돌려 기존 차이인지 확인한다 |
| D4 | 27B adapter image section | `mmproj-BF16.gguf` 파일이 필요하다(모델 폴더 확인) |
| D5 | C3 hash를 SHF_ALLOC section만으로 정밀하게 | 지금은 줄 번호만 바뀌어도 캐시가 무효가 된다. 안전한 쪽이라 낮다 |
| D6 | C1 테스트 보강(kv_only가 둘 이상 섞인 cohort) | CPU/GPU session test 추가 |
| D7 | cand4 배포 | 사용자 결정. 배포하면 supervisor(C2, `prepare_memory`)를 같이 설치해야 한다 |

## GPU 패스 구성 (한 번에)

1. 빌드는 GPU 측정 전에 끝낸다. 측정하는 동안에는 CPU 빌드를 하지 않는다.
2. drop+compact를 하고 buddyinfo를 기록한다.
3. CLEANUP 확인: adapter test, KVarN 취소 경로, C4 시나리오, logprobs 63행 thread 수.
4. S2/S3: batch를 cand2와 새 후보로 번갈아 5번씩 돌린다.
5. S4: `cache` disk 행의 pass 로그를 본다.
6. S1 (1): 재부팅하지 않은 상태로 MTP를 잰다.
7. S1 (2): 시스템 업데이트와 재부팅을 한다(2026-11-09 사용자 결정). 업데이트 전에
   kernel, linux-firmware, MES firmware 버전, `amdgpu` 모듈 인자(cwsr_enable,
   queue_preemption_timeout_ms)를 기록하고, 업데이트 뒤에도 같은 것을 기록한다.
   그 뒤 base와 새 후보로 MTP d0/d32K를 3번씩 잰다. 업데이트와 재부팅을 같이 하므로,
   속도가 돌아와도 둘 중 무엇이 원인인지는 가리지 못한다. 그래도 S1을 닫기에는
   충분하다.
8. 업데이트로 firmware나 kernel이 바뀌었으면 U4-3 hang을 다시 확인한다. cand2
   FP16 hang loop를 poke 없이 돌려, 우회 없이도 멈추지 않는지 본다. 멈추지 않으면
   strided grid를 계속 둘지 다시 정한다.
9. 모든 결과를 KNOWHOW에 기록한다.
