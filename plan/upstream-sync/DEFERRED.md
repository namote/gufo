# 미뤄 둔 일 계획 (2026-11-09, U6 뒤)

순서: CLEANUP 남은 항목(B1 KVarN, C4, C8a 동시 실행 수 제한) 구현과 리뷰 →
아래 GPU 패스 한 번 → 결과를 보고 고친다. 느려진 것 찾기를 먼저 한다.
운영 서버는 사용자가 말할 때까지 꺼 둔다.

## 1. 느려진 것 찾기 (먼저)

| # | 무엇 | 지금 아는 것 | 확인 방법 |
|---|---|---|---|
| S1 | MTP가 11/08 qualification보다 약 10% 낮다 | base `166953f0`(같은 binary, 같은 작업과 hash): 11/08 65.7/104.4, U6 drop+compact 뒤 58.9–59.3/94–99 tok/s(d0/d32K). 단편화로 39/68까지 떨어졌다가 대부분 돌아왔다. 남은 10%의 원인은 모른다 | (1) `compaction_proactiveness=50`을 적용한 지금 상태로 base MTP d0/d32K를 drop+compact 뒤 3번 잰다. (2) 재부팅 뒤 같은 측정을 한다(**재부팅은 사용자 승인이 필요하다**). 재부팅해서 65대로 돌아오면 환경 문제다(단편화, THP, page cache). 그러면 hugepage를 미리 잡을지 정한다. 돌아오지 않으면 11/08 이후 바뀐 것(firmware, kernel 모듈, docker image, HC session의 pack 변경)을 하나씩 대조한다. /proc/buddyinfo, THP 상태(`/sys/kernel/mm/transparent_hugepage`), GPU clock을 같이 기록한다 **U7(재부팅 전, 11/09)**: 기록은 `build/b1/u7-20261109/environment-before-update.txt`(kernel 7.0.0-38, 7.0.0-42는 설치만 됨, linux-firmware 20260319.git217ca6e4, MES 0x86, cwsr 1, preemption 9000 ms). **`compaction_proactiveness=50`이 MTP를 약 18% 늦춘다**: 같은 base·같은 work로 50/0을 번갈아 재면 50에서 d0 45.8~48.1(반복 사이 ±4~7), d32K 81.7~86.7, 0에서 d0 57.3~57.8(±0.3~1.1), d32K 89.9~93.0이다. AR은 그대로다(32.85/32.55). buddyinfo는 두 상태 모두 order-10 약 17.5K로 정상이고 측정 중 compaction·migration은 0이라 기전은 모른다. 0에서도 11/08보다 d0 약 12% 낮다(57.5 대 65.7). cand5는 0에서 d0 중립(57.5 대 57.5), d32K +14%(U4-3). **사용자 결정 필요**: `/etc/sysctl.d/60-gufo-compaction.conf`를 0으로 되돌릴지(이번 패스는 runtime만 바꿨다가 50으로 되돌렸다). 재부팅 뒤 측정은 같은 proactiveness 값으로 해야 비교된다 **U7b(11/10)**: 사용자가 2026-11-10에 `/etc/sysctl.d/60-gufo-compaction.conf`를 지워 proactiveness를 기본값 20으로 되돌렸다(이번 패스에서 20 확인, 바꾸지 않음). 재부팅 전 MTP 기준 측정은 하지 않았다: 22일 가동한 host의 disk 상태·page cache·단편화를 통제할 수 없어서, 사용자가 작은 차이 timing(S1·S3)을 모두 업데이트·재부팅 뒤 scripted preflight 아래로 옮겼다. 재부팅 뒤 측정도 20에서 한다 |
| S2 | batch pair 행 f64690(`batch_c2_greedy_short_pairs_match_serial`)이 cand4에서 wall 232→379 ms, decode 92→239 ms | cand4 한 번만 나왔다. cand2는 223.7 ms, fb→fc2도 같았다. residents 게이트(`6e2c504e`)는 두 요청이 같이 있으면 wide를 끄지만, base에는 wide가 없어 이것만으로는 base보다 느린 이유가 안 된다 | cand2와 cand4를 번갈아 5번씩 `batch`를 돌린다(drop+compact 뒤). 재현되면 그 행만 `GUFO_PROFILE`/serve 로그로 pass별 시간을 보고, residents 수가 바뀌는 순간(한쪽이 끝날 때) graph 재녹화나 scratch 할당이 생기는지 본다 **U7**: 재현되지 않았다. decode가 cand2 90.7~138.0 ms(5회), cand5 90.7~136.6 ms(8회), cand4 102~147 ms(3회), base 92.3이다. 모든 binary에서 약 91 또는 약 135 ms 두 값으로 갈린다. cand4의 239 ms는 한 번만 나온 값으로 보고 닫는다. pass별 로그는 모으지 않았다 |
| S3 | cand4 batch replies 행 +5~+25 ms(57개 가운데 14개 flag) | 합계는 base보다 빠르다(6,932 대 7,269 ms). 행별 5%/10 ms 기준에는 걸린다 | S2와 같은 반복에서 같이 본다. 반복해도 같은 행이 걸리면 S2처럼 pass별로 본다. 아니면 판단 보류로 기록한다 **U7**: 짧은 동시 행 24개 decode 합계가 cand2 2633~2755 ms(5회), cand4 2756~2823(3회), cand5 2769~3070(8회)이다. cand5의 모든 run이 cand2의 모든 run보다 크다(중앙값 +8.7%). base rpb 대비 flag는 cand2 6~8개, cand5 8~16개다. draft round는 24개 중 3개만 다르다. proactiveness 50(잡음이 큰 상태)에서 쟀으므로 판단 보류지만 방향은 일관된다. 의심: C4의 decode step마다 hipEventRecord, cand3/cand4의 strided grid·residents 게이트. 0에서 cand2·cand4·cand5를 다시 번갈아 재야 가린다 **U7b(11/10)**: 돌리지 않았다(S1과 같은 이유로 업데이트·재부팅 뒤로 미룸). 다시 잴 때 비교할 binary는 cand2·cand6(`c820c722`)·cand7(`d6a8c242`, C4 수정)이고, cand6이 cand2보다 모든 run에서 느리면 cand4로 원인을 가른다. 행 합계 스크립트는 `build/b1/u7b-20261110/tools/s3_sum.py`(U7 합계를 그대로 재현함) |
| S4 | `*-disk` 복원 prefill +10~15 ms(B3) | U5 후보부터 있고 M1 탓은 아니다. CLEANUP B3는 정렬 tail 때문에 kv_pool에서 pass가 하나 더 나뉘는 것으로 본다. U6 보고는 "원인 모름"이다 | 해당 행의 serve 로그에서 prefill pass 수와 경계를 base와 비교해 원인부터 확정한다. 확정되면 B3의 2번(짧은 정렬 tail은 capture하지 않음) 후보를 만들어 `pi_agent.py`(tool loop)와 일반 chat(thinking을 지운 history) 두 workload로 잰다. 둘 다 나빠지지 않을 때만 채택한다 **U7: 원인 확정(B3)**. cand5는 끝이 4의 배수가 아닌 `*-disk` 행마다 prefill이 끝나기 전에 정렬 경계 checkpoint를 하나 더 만든다(`persistence_refused tokens=184/186, 216/217, 708/709, 740/743, 708/711`, `cache_captures=2`, base는 1). 복원도 정렬 경계에서 한다(172 대 173, 668 대 670, prefill 1~2 token 증가). token 수가 같은 행에서 prefill +13~14 ms, snapshot 3.2→5.2 ms다. 끝이 정렬된 736 행은 추가 capture가 없고 +6 ms다. B3 2번 후보는 만들지 않았다(`s4-disk-rows.txt`) |
| S5 | C4: 다른 세션의 긴 prefill 동안 capture가 7~10 ms에서 1.83 s로 늘어난다 | CLEANUP 구현이 진행 중이다 | U6과 같은 `--sessions 2` 시나리오로 capture 시간과 TTFT를 잰다 **U7: 고쳐지지 않았다**. cand5에서 A capture는 혼자 9.3~9.6 ms이고, B가 약 24K token을 cold prefill하는 동안에는 1827~1834 ms다. A TTFT는 1.94~2.01 s에서 4.17~4.33 s로 늘어난다(U6 cand2 1829~1841 ms). event 대기를 넣어도 같다. 남은 대기는 `RetainRows`의 stream drain이 아닌 다른 곳에 있거나, 공유 executor stream에서 A의 event보다 앞에 들어간 B의 작업 때문이다. pass 단위 profile이 필요하다 **U7b: 원인 확정, 수정함**. `RetainRows` 대기가 아니다(50 ms 넘는 호출 없음). (1) scheduler가 비동기 capture를 시작한 요청을 바로 `capturing`에 세워 두고 B의 512 token pass(약 455 ms)를 돌리며, capture 완료는 pass 사이에서만 본다. A는 capture가 4번(2048 경계, 정렬 prefix, prompt 끝)이라 4×455 ms=1.82 s를 기다린다. (2) `GPU_MAX_HW_QUEUES=2`라 capture copy가 B pass 동안 진행되지 않는다(capture 하나 약 445 ms; 4로 올리면 2~8 ms지만 snapshot은 여전히 1.40 s). 수정(cand7 `d6a8c242`): step 뒤 capture를 최대 50 ms(`capture_settle_wait`) 기다린 뒤에만 세워 두고, prompt capture 시간은 worker가 끝난 시점까지만 잰다. A capture 1817~1832 ms→7.1~9.2 ms, A TTFT 4.09~4.18 s→3.18~3.29 s, B TTFT 그대로(21.1~21.3 s). 남은 TTFT 차이는 B chunk와의 prefill 순서다 |

## 2. 그 밖에 미룬 것 (느려진 것 다음)

| # | 무엇 | 메모 |
|---|---|---|
| D1 | U4-3 E1~E4 replay | hang 재현 도구. 우회가 통하므로 급하지 않다. `u6-rocgdb-nix` docker volume을 남겨 뒀다 |
| D2 | AMD에 MES queue 정지 버그 보고 초안 | 외부 공개라 사용자가 원할 때만 한다. 증거: `build/b1/u6-20261109/hang/` |
| D3 | `mtp_audit` base 대조 | 모든 후보에서 attention-state 0.0071(기준 0.002)로 실패한다. base에서 한 번 돌려 기존 차이인지 확인한다 |
| D4 | 27B adapter image section | `mmproj-BF16.gguf` 파일이 필요하다(모델 폴더 확인) |
| D5 | C3 hash를 SHF_ALLOC section만으로 정밀하게 | 지금은 줄 번호만 바뀌어도 캐시가 무효가 된다. 안전한 쪽이라 낮다 |
| D6 | C1 테스트 보강(kv_only가 둘 이상 섞인 cohort) | CPU/GPU session test 추가 **완료**: `1d6faa94`/`18081f1c`, U7b cand6 session test에서 `batch mixed_kv_only_cohorts=5` |
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

U7(2026-11-09 22:53~23:50)에서 1~6단계를 했다. 증거는 `build/b1/u7-20261109/`에, 수치는 KNOWHOW "Upstream U7 pre-reboot GPU pass"에 있다. 7단계(업데이트·재부팅) 전에 proactiveness를 정해야 한다(S1).

U7b(2026-11-10 00:00~01:00): cand6 빌드·format, C1 session test, C4 진단과 수정(cand7)을 했다. S3와 MTP 기준은 업데이트·재부팅 뒤로 미뤘다. 수치는 KNOWHOW "Upstream U7b pre-reboot follow-up"에 있다.
