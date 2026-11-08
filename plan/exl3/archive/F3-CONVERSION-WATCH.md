# c03 변환 시간별 점검

**현재 상태 (2026-11-07):** c03 본체·보정 MTP·원본 BF16 비전 조립과 최종 검증이 완료됐다. 사용자 승인으로 c03를 상시 서비스 중이며 변환 감시는 종료했다. 아래는 날짜별 계획·실행 이력이다. [최종 결과](F3-MTP-CALIBRATION.md).

2026-11-05 사용자: “이제 시작하자 한시간마다 확인해줘 잘안됐으면 수정하고 계속 돌리고”.
기존 중단 지시를 해제하고8K 소형 검사→본체 변환→오류 진단/수정/재개와 시간별 점검을 승인했다.
ngram은 완료본 재사용, 본체는 c03에서 새로 시작한다. 사용자 후속 중지·변경이 우선이다.

- task: `/home/m2on/projects/ai.tools/test/experiments/flash_next_quant_20261104`
- output: `/home/m2on/projects/ai.tools/tmp/test-results/flash-next-quant-20261104`
- 데이터/준비 내역: [c03](F3-LONG-CALIBRATION.md), [비트 배분](F3-QUANT-BIT-ALLOCATION.md).
- supervisor: `model_job.py`; 새 기록 `model-job-c03/STATE.json`, 모듈별 `module-*/run.log`.
- pilot: `long_row_job.py` / `long_row_pilot.py`, 최대1,200초. c03의8192/512 행을
  첫GDN/PLE/첫full-attention에 통과시키고 all-expert H·normal routed state·B1 packed 연산 확인.
  이전512 baseline은 재실행하지 않는다. 통과 영수증과 코드/data hash를 main 시작 전에 검사한다.
- main unit: `gufo-quant-model-c03.service`; 시간별 점검 timer: `gufo-quant-hourly.timer`.
  `WATCH-CONTROL.json`이 점검 활성 상태를 보유하며 `hourly-watch/`에 전송 결과를 남긴다.
  local systemd timer가 `codex queue --resume`로 같은 채팅에 점검을 보낸다.
  Codex 앱의 Scheduled 항목을 만든 것은 아니다. 머신/로컬 daemon 가동이 필요하다.

## 매회 행동

## 사용자 추가 점검: 레이어별 출력 오차 추세

시간별 점검 때 기존 완료 로그의 rfn/cos/sqnr도 읽어 급변·비정상값을 확인한다.
별도 GPU 실행 없이 기존 자료만 재사용한다. 현재 지표는 첫5 calibration 참조행에서
동일한 현재입력을 원본/양자화 블록에 넣은 출력 비교이며, 독립holdout 또는
전체원본모델 대비 누적품질손실/정답률 하락이 아니다. 마지막 전체검증은 예정대로 묶는다.
2026-11-06 layer3~42 관측: RFN 0.007205~0.030409, layer36 최대 뒤 layer42 0.019864.
후반이 초반보다 높은 경향은 있으나 계속 증가하지 않는다. layer38 SQNR 최저38.728951dB,
layer41 39.208468dB는 최종검증 관심 대상으로 기록하며 이 수치만으로 실패판정하지 않는다.
근거 hourly-watch/LAYER-ERRORS-1793950471.json .

## 최신 사용자 승인: MTP 전체 캘리브레이션 보정 추가

**Gufo 입력 보정 후보 CPU 준비 완료 / 실제 GPU 검증 대기.**
pinned 변환기의 per-stream norm/zero-first 입력과 Gufo 실행의 차이를 확인하고,
별도 후보를 F32 전체 HC*H RMSNorm 및 tokens[1:]+hidden[:-1]로 수정했다.
전체 2309문서 유지, 유효 쌍 1997187개/hidden FC H 표본 7988748개다.
CPU 10그룹과 독립 리뷰 통과, 후보 6파일 및 영수증 SHA 재확인 완료.
근거 mtp-calibration-candidate/{CPU,GUFO-CPU-REVIEW,GUFO-SEMANTICS-REVIEW}.json.
실제 native FC/packed codec와 T-1 길이 Lt 1510 shape 자격 검사는 아직 대기다.
CPU 통과를 전체 보정/품질/속도 통과로 취급하지 않는다. 이전 Kyojin 후보 리뷰는 대체됐다.
idx51 예약은 유효하며 실행 중 본체 및 동결 코드는 수정하지 않았다.

최신추가지시(2026-11-06): FP32/FP16도최종체감속도와초안정확도절충으로판단.
초안정확도가낮아도최종token/s가높으면채택후보,본문검증정확성은별개로유지.
통계수집속도와실제추론속도를구분하고검증은최종창에묶는다.

최신사용자: MTP는속도민감. 수락률단독아닌draft+검증+거절복구포함최종token/s로판단.
동일완료본체에서MTPoff/K4기준/혼합후보를최종창에묶어비교하며저장baseline조건확인.
후보에 H/refs 영속저장 및 재개 구현·CPU검사 완료;입력FC비트변경시decoderH입력도변하므로무조건재사용금지.
혼합후보이론가중치증분20.703125MiB이나매토큰dense비용은실측필수.

예약 helper CPU7/독립리뷰통과,gufo-mtp-calibration-boundary.service active.
현재waiting_for_layer47: 본체실행/락에는아직관여하지않으며Layer47실제실행때만GPU락대기.
idx51에서2309행/F32 HC4x2560/1999496토큰/cleanup검사후예약·같은채팅callback.
알림전송실패도락을풀지않고시간별점검으로이어간다.예약실패/본체디스크중단시
원인과기존BOUNDARY보존후명시재예약해야하며,main만재개해서idx51을지나치지않는다.
근거mtp-calibration-boundary/{REVIEW,ACTIVATION,BOUNDARY}.json.

사용자 “캘리브레이션 보정도 하자” 승인. 기존uncalibrated MTP 계획은대체된다.
[F3-MTP-CALIBRATION.md](F3-MTP-CALIBRATION.md)대로 본체idx51 pre-mixer 상태를
읽어전체2309행1999496토큰 MTP입력결합+1층 보정·혼합비트·비전BF16을준비한다.
현재본체정상실행은유지한다. mtp-calibration-boundary/BOUNDARY가
waiting_for_layer47/waiting_natural_boundary/reserved이면 예약을중복실행하지않는다.
**reserved이면 main을재개하거나RELEASE하지말고 MTP검사/보정/최종연결부터처리한다.**
본체idx51에서먼저MTP를처리하고,이후mixer/head로이어간다.기존완료본체재계산금지.
입력fc두개qmap분리/shifted hidden/4(T-1)카운터가필수이며후보는아직GPU미검증이다.
단순recipe추가만으로mtp_bits일괄K4를바꿀수없으므로실제적용값검사가필수다.

## 최신 사용자 방향: MTP도 본체처럼 역할별 혼합 비트

사용자는 MTP 일괄K4보다 본체와 유사한 역할별 혼합배분을 선호한다.
실제 b1 recipe의 full-attention layer3 매핑을 mtp.layers.0로 옮긴1544항목과
입력결합 fc_hidden/fc_embedding K8 두항목, 총1546개 후보를 별도 저장했다.
전문가 gate/up/down K4,공유전문가 K6,q/o K6,k/v/indexer K8,입력결합 K8.
본체 routed expert는 현재 전부K4이며 전문가별 자동민감도 배분을 완료한 것은 아니다.
MTP-MIXED-BITS-PROPOSAL.json은 검토/통합 전 후보이고 현재실행은 mtp_bits4 그대로다.
비전BF16 변경과 같은 자연경계에서 recipe/side-model 실제적용경로·누락/중복·
최종 metadata/동결을 검토해 최소통합한다. 실행중코드수정/본체재양자화금지.
기존MTP는uncalibrated경로임을유지하며 혼합비트만으로본체와같은품질이라주장하지않는다.

## 최신 사용자 변경: 비전 원본 BF16 유지 (후속 변환 반영 대기)

사용자가 비전 원본 BF16 유지를 선택했다. 원본 model.visual.* 333 tensors,
448931056 parameters/897862112 bytes(0.8362GiB), 연결부 포함.
현재 실행 중인 convert_main.py는 아직 --vision_bits 8 및 vision_bits=8 기록을
사용한다. **문서 변경만으로 런타임 반영된 것이 아니다. 최종 side-model 변환 전에
자연 모듈경계에서 BF16 비전 보존/최종팩 연결 및 로더 호환성을 검토하고,
최소 구현·CPU검사·독립리뷰·이전동결보존/새동결을 완료해야 한다.**
실행 중 코드나 frozen recipe를 수정하지 않는다. 현재 정상 본체는 유지한다.
이 절의 초기 MTP K4 계획은 위 혼합 비트·전체 보정 승인으로 대체됐다. 실행 인자는 아직 변경 전이다.

## 추가 요청: K4V4 비교 준비 (완료된 변환과 별도)

사용자2026-11-07 “우리도 k4v4로 바꿔보자”. 같은c03팩/동일Nix10.1실행기의
KV만kvarn-k4v4로바꿔저장FP16품질/성능자료와비교한다. 기존변환·FP16baseline
재실행금지. hourly timer와WATCH-CONTROL은완료/비활성상태그대로유지한다.
별도 final-validation/kv4/CONTROL.json으로이번검사만제어,콜백알림생성없음.
품질32760위치·MTPoff/on속도·C1/C2/cache확인을묶어직렬검증한다.
모델재양자화/서비스자동배포없음. 새검증helper만별도준비/CPU/독립리뷰후실행.

## 완료: c03 조립·측정 범위 검증 / 감시 종료

2026-11-07 사용자 “조립후검증” 완료. 시간별 timer disable/stop,
WATCH-CONTROL.enabled=false. 완료본체idx53과MTP/H/refs/idx51/idx52/기존baseline/
Nix10.1중간빌드/최종팩보존. 변환본체·과거실패모듈을재시작하지않는다.
최종근거 output/FINAL-QUALIFICATION.json(status=qualified_for_measured_scope).
상시서비스자동실행/기본모델변경없음,검증용서버는모두정리됐다.

최종팩: output/mtp-final-assembly/data/final001/pack.
실행기: task/final-current-nix101/runtime/gufo(6edf0d44…),동일빌드collectorbc989832…,
source a81c0f93/NixROCm10.1. RUNTIME.json/compile영수증은검사시점고정으로유지,
후속FINAL-QUALIFICATION이완료상태를보유한다. 구e7이미지내ELF는K8미지원이므로
최종팩실행에재사용하지않는다. 이미지환경+새ELF+/nix volume의조합이검증대상이다.

품질: 저장HF BF16의32760위치재사용,64head replay exact,finite/cleanup통과.
NLL2.162650(BF162.172846),KL2560.041522,top1일치90.8394%.
문서/code/합성대화8창이며독립holdout미입증,정답률또는전체어휘KL이아니다.
기존UD와top1차이는7위치,과거runtime차이를양자화효과로단정하지않는다.

정확성: HTTP10요청MTPoff/on본문/종료/토큰수동일,산술/비전/이력/캐시통과.
32토큰smoke와TG128 depth0/32768각2회최종tokenSHA동일,
HTTP4greedy요청C1/C2 off/on모두동일. 원래smoke003로그혼입파싱실패는보존,
supplement001에서미실행MTP만추가하고저장off재사용했다.

속도(off→MTP최대7,기존adaptive정책): TG0 27.33→50.70tok/s,
TG32K 26.72→87.30(해당샘플100%초안수락),PP2048 1460.97→1370.97,
PP8192 1501.37→1403.43. MTP프리필약6–7%비용관측.
HTTPgreedy4요청전체wall기준C1 26.13→41.70,C2합산36.56→45.55tok/s.
모든8case성공/cleanup0/KFD0/팩불변,측정중최소가용39.65GiB.
순차모드순서/적은반복/희소온도표본한계,32K검증을full256K/512K자격으로확대금지.
K4일괄MTP/다른FP16·FP32초안연산비교와GPUzero-copy소유권증명은미실시.
상세: final-validation/quality-current-001/QUALITY-RESULT-REVIEW.json,
final-validation/throughput-001/{RESULTS,RESULT-REVIEW}.json.

## 이전 상태: 팩 완료 / K8 지원 Nix10.1 검증 런타임 준비

2026-11-07 사용자 최종검증 승인. smoke001은 CLI의 thinkoff+reasoningmedium
충돌로 모델로드전종료. DFlash2전용 draft-policy도 MTP에서제거하고 실제 장치없는
4개 CLI검사 뒤002 실행. smoke002는 Flash-Next EXL3 lm_head geometry/rate/codebook
gate에서종료, cleanup0/packunchanged. 완료팩의 lm_head는정상K8 mul1,
I16[160,15520,128], suh2560/svh248320이며 config/tokenizer padding248320일치.
실제검증이미지e7a761a40187의옛ELF는K3..7loader와K1..7GPU GEMV만지원한다.
조건완화만으로우회금지: 실제ELF K8 kernel0, 현재소스2095529d에는K8구현있음.
현재HEAD coherent Nix/ROCm10.1 실행기+품질collector 준비중. 기존refactor-gpu는
7.2.4빌드이므로사용자10.1요구에맞는대체로사용하지않는다. 모델재양자화/metadata왜곡금지.
완성본체idx53/혼합MTP/비전/idx51/idx52/원본/기존참조보존, main서비스재시작금지.
검증failed는현재옛실행기호환성실패이며새팩품질수치실패가아니다.
근거 final-validation/smoke-{001,002}/; mtp-final-validation-preparation/loader-diagnosis/.
실제새runtime준비후새동결·독립리뷰·작은로드검사→품질·성능단일창으로이어간다.
시간별watch는변환재개를하지말고이runtime준비/검증작업진행을확인한다.

## 이전 상태: 조립 완료 / 최종 검증 준비

2026-11-07 final001 조립159.31초 완료. MTP1546개 혼합투영 payload exact,
비전333개 BF16 897862112B payload exact, source_artifacts_unchanged=true.
감독cleanup오류0/KFD0/protected_unchanged=true. 서비스자동실행없음.
사용자 “오케이 진행하자 조립후 검증”에 따라 같은production image/Nix10.1에서
텍스트·비전·MTP off/on 정확성 및실제속도, 기존BF1632,760위치참조 품질검사를 준비한다.
참조101424960B SHA de728d88… 검증완료, 재생성금지. 참조검증영수증은
mtp-final-validation-preparation/REFERENCE-PAYLOAD-VERIFIED.json.
원본BF16용 EXPECTED-SHA256/VERIFY-SHA256/PROVENANCE 문서3개를
팩내 source-provenance/로원문보존이동하여 새EXL3 shard검증문서로 오인하지않도록함.
근거 mtp-final-assembly/PROVENANCE-LABEL.json. 텐서와모델설정변경없음.
미사용24시간이전Docker buildcache만정리, 루트여유127.86→약187GiB.
모델/idx51/idx52/H/packed/baseline보존. 검증하네스 CPU/독립리뷰전GPU실행금지.
현재완료는조립까지이며 최종품질·MTP최종token/s는미검증. main재시작금지, 시간별감시유지.

## 이전 실행: 본체 완료 idx53 / 최종 팩 조립 final001

2026-11-07 head recovery001 독립 결과 리뷰 통과. 첫5 원본-vs-K8 전체행은
모두 finite, 평균 RFN0.00386195(0.3862%), 범위0.2734~0.5000%, 평균SQNR48.5118dB.
소요554.55초, cleanup오류0/KFD0, 보호metadata7171개 불변. 해당5 calibration
입력의 출력층 지표이며 전체모델 품질/holdout/최종추론속도 결과가 아니다.
저장된5로그잇 payload SHA를 CPU승격 시 확인하여 idx53 확정 완료.
첫5 packed로그잇+2304원래prehead입력을 hardlink한 terminal형식 유지,
원래idx52/독립idx51/완료K8와 module062 실패 영수증 보존. H/양자화 반복 없음.
근거 head-recovery/{RESULT-REVIEW,COMMIT-JOURNAL}.json; journal phase=complete.
MAIN-CONVERSION=main_complete_side_models_skipped; 원래본체서비스 재시작금지.

검토된 최종조립 계약506pins/실제main53/MTP7영수증/보존idx51/혼합1546개를 확인,
기존 pending 계약을 별도 보존하고 부모 승인. gufo-final-assembly-001.service
PID1366613 정상실행, final001 private출력만 쓰며 원본/본체/MTP는읽기전용.
기존ngram은같은inode hardlink, calibrated혼합MTP와333개BF16비전 payload동일검사.
시작디스크191.87GiB/필요171.89GiB(100GiB여유포함), RAM110.78GiB.
1200초/8GiBRAM/100GiBdisk 하한 감시; 서비스배포/새양자화없음.
영수증 mtp-final-assembly/runs/final001/{STATE.json,run.log};
출력 mtp-final-assembly/data/final001/pack/GUFO-COMPILE.json.
완료callback후 cleanup/원본보호/최종팩을 독립검토하고 마지막 정확성·성능을 묶어서 검증.
조립완료만으로품질/성능통과처리금지. 시간별감시는 최종검증완료까지 유지.

## 이전 실행: 출력층 저장 오류 복구 recovery001

2026-11-07 module062는전체2309행H수집2405.63초와K8양자화637.64초를완료한뒤,
첫5정상참조forward의row1(7793토큰)3.60GiB로그잇을2GiB비동기쓰기큐에넣다
ValueError로중단됐다. 양자화실패/GPU오류/메모리부족아님. cleanup0/KFD0,
원본idx52/bad_rows[]와완료lm_head636201289B/실제K8/finite scales보존.
head pack SHA4fce99813938a2ed9ee47cc4cdf4d8d68a508c31ecfde883c1a3c44c0cd564f7.
원래52도head-recovery/protected-idx52/ckpt에immutable행hardlink+독립metadata로보존.
본체48층/mixer/MTP/H/packed/ngram/idx51보존. 기존head통계·양자화재실행금지.

별도head_recovery.py는원본및완료K8로첫5 full-row출력만재비교하며CPU원래
get_state_error reduction/원래dtype·apply_out_scales=True유지. 사라진RAM참조만복구한다.
5quant로그잇총9.18GiB를private파일로직접저장해async2GiB큐를우회한다.
나머지2304행은원래input을유지하는정상terminalcheckpoint형식을보존한다.
H/quant/main호출명시금지, 기존완료ROW는영수증검증후재사용가능(실행재개는별도검토).
후보CPU5/독립리뷰,감독CPU7/부모리뷰,infra43218.73초/실제sourceadmission통과.

unit gufo-head-recovery-001.service,runner PID1362525,현재별도GPU작업실행중.
예산150GiBstart/100runtime,28GiBRAMstart/8runtime,1200초/ownedcleanup/완료callback.
원본/모델/본체/MTP는RO,출력head-recovery/data/recovery001/result만privateRW.
실행중후보/감독/계약/동결파일수정금지. 기존mainunit는실패상태로유지하고재시작금지.
영수증 head-recovery/runs/recovery001/STATE.json; 계약head-recovery/contracts/recovery001.json.

완료시5ROW finite/오차/로그잇SHA·헤더/cleanup/보호입력을독립검토한다.
그뒤별도CPU head_recovery_commit.py(CPU8/COMMIT-REVIEW독립통과)가첫5로그잇+
2304원본행을hardlink한idx53을원자journal로승격하고MAIN marker/compileargs를쓴다.
module062실패기록은변경하지않고복구영수증으로연결한다. GPU래퍼는자동승격하지않는다.
CPUcommit 중단시journal을검토해rename단계부터복구;완료된head GPU재계산금지.
승격실행전결과review JSON에status=head_recovery_reviewed_for_idx53,
commit_sha256(현재5ddce6498436bba90ddab53229c07fd2e96c0314408d2d8baf9fae2e18fd0a29),
result(호스트경로),audit_sha256,runner_state/runner_state_sha256,summary_sha256,
failed_module_state_sha256를묶고 `head_recovery_commit.py --review <결과review>`로실행.
runner 성공상태는completed_pending_independent_review이며summarySHA까지일치해야한다.
helper는두실제락/KFD0/RAM8GiB/disk100GiB를확인하고메타데이터승격만한다.
실행중인복구결과는아직pending이므로이CPU준비리뷰를결과통과로오인하지않는다.
승격뒤아래prepare_final_assembly_contract.py/별도조립감독기로이어가며
전체정확성·성능은마지막묶음검증까지pending.감시는유지한다.
근거 head-recovery/{AUDIT,CANDIDATE-CPU,CANDIDATE-REVIEW,RUNNER-CPU,RUNNER-REVIEW}.json.

## 이전 점검: mixer 완료 / head 실행 / 다음 조립 절차

시간별점검 2026-11-06 23:16 KST: module062 정상실행, 로그가4→76/2309행으로
전진했다(phase505.64초). 가용RAM81.02GiB/내장201.74GiB, 상태갱신1초미만,
최근15분 kernel GPU timeout/reset/OOM 일치로그없음. KFD는현재변환PID1356867 하나.
idx52/bad_rows[]·보존idx51정상, 코드/프로세스재시작/추가GPU없이유지.
근거 hourly-watch/CHECK-1793974592.json. 최초행시간의전체외삽은계속보류한다.

module061 mixer2309행완료·idx52, module062 head 캡처실행중. 새head는첫3행
3217/7793/4655길이에서70.25/237.69/339.46초전진; 메모리부족/수치오류없음.
head입력·weight·출력FP16/누적FP32이며 ext.hgemm dtune은새Mclass마다
rocBLAS여러후보·정확성·반복시간탐색을수행한다. 현재첫길이는서로다른class,
전체22class중2151행은512class. 초기시간을200만토큰전체에직선외삽하지말것.
자동탐색기여는코드기반원인후보로아직분리실측아님; 정상module062유지/실행코드수정금지.

본체가main_complete_pending_calibrated_mtp_vision_compile로종료하면 실제idx53,
module061/062성공·cleanup0/KFD0/원본보존/MAIN-CONVERSION·MAIN-COMPILE-ARGS를
독립검토한다. 완료head재실행금지. 다음CPU준비기:
`test/experiments/flash_next_quant_20261104/prepare_final_assembly_contract.py`
(PYTHONPATH에는해당task와ai.tools루트). 실제완료전에는draft를만들지않는다.
준비기는 `mtp-final-assembly-runner/assembly-final001.json` 및
`mtp-final-assembly-runner/contracts/final001.json`을pending으로만쓴다.
부모검토후 assembly status=approved_for_final_mtp_vision_compile,
계약identity의assemblySHA갱신, 계약status=reviewed_for_final_assembly,
계약REVIEW status/contract_sha256를기록한뒤순차실행한다.
감독기 `mtp_final_compile_job.py --contract <위실제계약>`은새GPU/JOB락,
원본RO/privateRW,완성팩+8GiBmargin+100GiB여유예산,28/8GiBRAM,
1200초/ownedcleanup/종료callback을제공한다. 기존idx51lease는이미released이며
mtp_calibration_job으로최종조립하지않는다. 후보코드CPU8/부모독립리뷰통과.
조립후서비스자동실행금지; 저장BF16reference32760행/a01를재사용하는최종검증준비는
`mtp-final-validation-preparation/{PLAN.md,EVIDENCE.json}`에있다.
같은본체K4·FP16/FP32 draft실행후보는아직별도준비가필요하며,다른팩숫자로대체하지않는다.
최종전체품질·실제token/s는미검증상태다.

## 현재 실행: 본체 마지막 mixer/head / 보정된 MTP 최종 조립 준비

2026-11-06 23:04 KST main-only converter/job을함께승격하고 idx51에서재개했다.
module061 mixer 실제1069/2309행 전진확인. 본체Layer0~47/MTP재계산없음.
이전467동결/두caller/STATE는repair-018-final-mtp-vision에보존했고 새470동결.
명시main-only디스크계약은idx51/52에서만예전실행되지않는compile예산을제외하며,
100GiB runtimefloor+실제state+16GiBmargin/원래cleanup은유지한다.
CPU8(예산4+완료4)/독립리뷰/통합전체CPUadmission/infra432(18.91초)통과.
예약RELEASE와GPU/JOB락반환확인후기존main unit재개. 실행중코드수정금지.

idx51은mtp-protected-idx51/ckpt에원본rowsidentity로별도보존됐다.
독립리뷰후완료derived2309행payload76.19GiB만퇴역,manifest/H/refs/packed/receipt보존.
free211.325GiB확보·fstrim완료. main은MAIN-CONVERSION/MAIN-COMPILE-ARGS를쓰고
idx53에서끝나며예전uncalibratedMTP/vision재변환·자동compile은실행하지않는다.
완료시새최종조립계약/디스크예산을검토해기존MTP·ngram+원본BF16비전으로조립한다.
별도조립감독기는CPU8/독립리뷰준비완료,아직조립실행없음. 모델서비스자동실행금지.
근거 model-job-c03/repair-018-final-mtp-vision/{MIGRATION,RELEASE-VERIFIED,ACTIVATION}.json,
mtp-final-compile-candidate/main-only-disk/REVIEW.json 및mtp-final-assembly-runner/REVIEW.json.

## 이전 완료: MTP 전체 보정 / 최종 연결 준비

2026-11-06 full-decoder-001 정상종료·독립리뷰 통과. 2309행/1997187쌍,
516H 모두해당카운터/finite/inf_nan0, actual1546투영 K4:1536/K6:5/K8:5.
전체1364.01초(통계907.01초/양자화424.12초/5정상참조5.58초), sourceunchanged/
cleanup0/KFD0. 첫5보정참조 RFN0.03711~0.05571/SQNR27.05~30.85dB;
독립holdout·전체모델품질·초안수락률 증명아님. 최종전체token/s선택기준 유지.

원본idx51을 mtp-protected-idx51/ckpt에독립checkpoint로보존했다.
immutable행hardlink+독립job/IDs,2309행/1999496토큰·기존rowsidentity정확일치,
새76GiB복사없음. 본체48층/입력H/디코더H/모든packed/refs/완료영수증보존.
파생상태76.19GiB는독립보존검사후행payload만퇴역가능하며manifest는남긴다.
이후FC비트후보에는보존idx51+해당FC로명시재생해야하며 기존decoderH무조건재사용금지.

본체idx51예약은아직유지. 검토된main-only converter/job을함께통합하고
불필요한옛side-model/compile를막아mixer/head부터이어간다.
head에서더이상실행하지않는옛compile디스크예산만별도엄격계약으로분리준비중.
최종조립은완료MTP재사용·비전BF16payload보존·ngram재사용·새출력으로진행한다.
근거 mtp-full-decoder-review/{REVIEW,CPU-AUDIT,LOG-AUDIT}.json,
mtp-protected-idx51/PRESERVATION.json, model-job-c03/repair-018-final-mtp-vision/.

## 이전 실행 완료: MTP 전체 디코더 보정 full-decoder-001

2026-11-06 22:29 KST full입력완료 독립리뷰 후 디코더 보정을 시작했다.
unit gufo-mtp-full-decoder-001.service, runner PID1347921, one-shot callback/Restart=no.
full-input-001은1441.10초 완료: 전체2309행/1997187쌍, hiddenH7988748,
embeddingH1997187. 저장H/refs/실제K8두FC finite·hash/2309행F32 T-1 header·inode
검사통과, GPUcleanup0/KFD0/원본idx51불변. 첫5보정참조 RFN0.01080~0.01382는
입력결합블록 오차이며 전체모델품질/초안수락률의 판정이 아니다.

같은 full/work의 입력H·packed·refs·derived76.19GiB를 재사용하며 입력 재계산 없음.
계약450pins/본체467동결/실제GPU·JOB예약lease 확인, 필요122.05GiB/여유139.58GiB.
실행 GPU admission original_quiet 통과. 전체516decoderH 각각1997187카운터,
5normalrefs/finite/실제혼합1546배분/cleanup을 완료callback에서 독립검토한다.
후속검증은 완료된 입력과 저장baseline을 재사용하며 25%후보는 적용하지 않는다.
원본idx51 예약은 유지하며 디코더 완료만으로 RELEASE/main재개하지 않는다.
최종조립/비전BF16보존/uncalibrated MTP차단 검토·통합 이후 mixer/head를 처리한다.
FP32/FP16·초안정확도 절충은 마지막 전체draft+verify+거절복구 token/s로 판단한다.
근거 mtp-full-input-review/{REVIEW,CPU-AUDIT}.json,
mtp-calibration-runner/contracts/full-decoder-001{,-review,-admission}.json,
mtp-calibration-runs/full-decoder-001/ 및 mtp-calibration-callback/full-decoder-001/.

## 이전 실행 완료: MTP 전체 입력 보정 full-input-001

2026-11-06 21:59 KST full입력보정을시작했다. 22:01 실제capture172/2309행전진확인.
단위 gufo-mtp-full-input-001.service, callback PID1343547/runner1343548,
재시작없음/ownedcleanup/종료시동일채팅알림. 실행중코드·full계약·CPU영수증수정금지.
데이터 /mtp-calibration-work/full/work, 감독 mtp-calibration-runs/full-input-001/,
알림감독 mtp-calibration-callback/full-input-001/. full입력은원본FP32BMM을사용한다.
전체2309문서/원본1999496토큰/유효1997187쌍, hiddenH7988748/embeddingH1997187.
25%미적용. mixedrecipe1546=K4 1536/K6 5/K8 5. 본체48층/idx51·예약락은그대로보호한다.

2행decoder002는1544양자화/전체516H카운터/finite/normalrefs/실제bitrecipe/cleanup통과.
출력RFN0.04105/0.02846은보정입력2문서의블록오차로전체품질/최종수락률아니다.
이자료와native raw/packed/Gram 검토로FULL-QUALIFIED.json 승인(전체품질/속도pending).
MTP terminal3텐서는원본BF16→F16반올림값과정확일치(maxabs2.98e-8);
비전333텐서의원본BF16보존계획과혼동하지않는다.

probecleanup후stagebudget runner만검토본으로통합/이전코드·영수증보존,
canonical CPU5+budget4/실제예약lease와full445pins 확인. 입력필요196.325GiB,
시작여유217GiB. 다음decoder는완료input-checkpoint/receipt/FCpack/2309행을검증한뒤
새76GiB할당을중복요구하지않고122.05GiB예산으로실행한다. 자동다음단계실행은없다.

입력완료callback에서input H/pack/derived를검토하고같은full work의decoder로이어간다.
새full-decoder계약에는 decoder_packed_upper_bound_bytes>=2GiB,
decoder_input_receipts_sha256(INPUTS/INPUT-COMPLETE/mtp.PACKED),
decoder_derived_state_identity가필수. 기존정상입력보정/H/packed재실행금지.
MTP완료전boundary RELEASE/main재개금지. 완료후최종조립후보/동결승격/본체mixer·head를
진행하고검증은마지막에묶는다. 본체원본idx51독립보존을먼저하고,
fullderived76.19GiB퇴역은실제완료증거검토후에만처리한다.
FP32/FP16·비트선택은최종draft+verify+거절복구포함token/s기준, 초안수락률단독판단금지.
근거 mtp-calibration-runs/full-input-001/ACTIVATION.json, mtp-stage-budget/INTEGRATION.json,
mtp-calibration-candidate/{FULL-QUALIFIED,FULL-QUALIFICATION-REVIEW,GRAM-FULL-QUALIFIED}.json.

## 현재 MTP 경계: 본체 Layer0~47 완료 / idx51 보호 중

module060 Layer47 저장성공, checkpoint idx51/2309행/1999496토큰/bad_rows[] 보존.
mtp-calibration-boundary는 reserved이며 pid1319760이 GPU/JOB락을 보유한다.
본체 supervisor의 BlockingIOError는 예정된 경계인계 결과이며 계산실패가 아니다.
새 작업은 예약lease를검증한별도runner만 사용하고 RELEASE/main재개 금지.
467동결·원본checkpoint stat/행header·GPUcleanup 확인, 가용110.88GiB/디스크221.46GiB.
최신 Gufo F32 fullHC/T-1 입력계약을 적용한다. callback의 zero-first/4T는 옛문구다.
MTP candidate CPU준비본을 native norm/FC/codec 및 shifted Lt연산 소형검사로
검증하고2행 단계별probe를재사용한뒤 전체2309행 보정으로이어간다.
현재 기존본체/동결/정밀도수정없음. 후보 및 runner는 별도파일로준비중.
근거 mtp-calibration-readiness/ADMISSION-INITIAL.json, boundary/BOUNDARY.json.

## MTP 자격 검사: native 통과 / 2행 decoder 검사 실행 중

2026-11-06 21:48 KST: native raw002/packed001 통과, source불변/cleanup0.
실제K8 codec와입력회전2FC exact, native대candidate 최대relative1.087e-6.
2행입력보정31초완료, H34808/8702·source정렬정상, refsRFN0.00941/0.00823.
input packed/H/derived/INPUTS 모두보존, 해당입력양자화반복없음.
Gram1510지원/20edge검사완료. 원래1e-5는긴FP16 5개실패유지,
기존8K자료와같은규모(max2.21e-5)·별도3e-5 후속screen통과. 전체품질통과아님.

raw001 readonlybind간EXDEV는BorrowedRows로수정(CPU5/실제RO연결검사/독립리뷰).
원본행복사/하드링크없음, input전체advance한private행만새checkpoint저장한다.
decoder001은계산전dynamicflush inspect.getsource 오류. 실제marshal코드+leafSHA로
신원기록수정, 명시migration으로기존INPUTS/packed/H를수정하지않고재사용.
새run_module이기존cm.main 자동scope와다르므로 index1의폐기출력capture에만
기존guarded _discarded_capture를명시연결(CPU7/독립리뷰). normalrefs/input경로불변.
현재 mtp-calibration-runs/probe-decoder-002 정상양자화중이며실행중코드수정금지.
실제2scope/2staticroute/emptydown1024/gatherreuse1022/countreuse2046확인.
저장H CPU감사516maps/count8702/infNan전부0, 2normalrefs보존.
최종decoder출력/bitrecipe/cleanup검토전전체보정은시작하지않는다.

본체48층/idx51은boundary reserved(pid1319760)로보호중. main재개/RELEASE금지.
다음: probe완료독립검토→별도fullinput/decoder와종료callback→최종통합/동결/재개.
실행관리기decoder디스크예산이76GiBderived를이중계산하지않도록별도후보준비중;
현재runner변경없고probe종료후CPU/리뷰/이전영수증보존뒤반영할것.
최종조립후보는CPU/독립리뷰준비완료이며생산미적용. 원본BF16비전333텐서보존,
옛uncalibrated MTP차단, 완료MTP의입력FC→derived→H→packed 연결검증을포함한다.
full완료후derived약76.19GiB퇴역은별도실제완료감사후판단, idx51원본은유지한다.
근거 mtp-calibration-runs/, mtp-borrowed-source/, mtp-final-compile-candidate/,
mtp-calibration-work/main/probe/{DECODER-CAPTURE-SCOPE,H-CPU-AUDIT}.json.

## 최신 시간별 점검: Layer45·46 완료 / 마지막 Layer47 실행 중

module058/059는 idx49/50 저장성공·bad_rows[]·cleanup0/KFD0.
Layer45 RFN0.014714/SQNR35.413249dB/1537.51초,
Layer46 RFN0.015587/SQNR36.378017dB/1976.68초.
RFN은 기존관측범위, SQNR은 이전최저보다낮아 최종검증 관심항목으로기록하되
이 지표만으로 실패나 전체품질하락으로 판정하지 않는다.
Layer46의 write_wait515.02초 기록은 이전층보다긴쓰기대기 증거이며
하드웨어원인이나 전체지연기여를 분리측정하지않았으므로 단정하지 않는다.
현재module060 Layer47은 통계수집완료/양자화로그계속갱신, 가용RAM약74GiB/
디스크222.71GiB. 정상본체는 유지, 코드/추가GPU검사/재시작없음.
MTP helper는 waiting_natural_boundary/observedidx50으로 GPU락대기중.
**다음 idx51 reserved는 예정된 보정경계다. main 재개/RELEASE보다 MTP검증·보정을 먼저한다.**
근거 최신hourly-watch/CHECK 및 module058~060/STATE·run.log.

## 최신 점검: Layer44 완료 / 퇴역 실패 임시입력 정리 후 idx48 재개

module057 Layer44 전체 계산1567.70초 정상완료, RFN0.019801/cos0.000312/
SQNR37.954798, idx48/2309행/1999496토큰/bad_rows[]/cleanup0/KFD0 확인.
다음시작 free148.93GiB < required192.275GiB여서 계산전 안전중단했다.
옛 실패 module056의 row-scratch가 Layer44 이전입력76.275GiB를 단독보유:
실패미복구 자료 보호정책상 자동정리되지 않은 것이 원인이다.
대체 module057 성공 뒤 독립2309행header/stat/inode검사로 현재idx48 겹침0/
전부nlink1/원본·baseline 참조없음 확인. 해당scratch만 삭제하고 실패로그/스택은 보존.
현재checkpoint 전후 inode/행수/크기 동일, 여유225.22GiB/다음필요192.275GiB 통과.
본체코드467동결·calibration·정밀도 불변, 완료Layer44 재계산 없음.
19:38 KST module058 Layer45를 idx48에서 재개했다. MTP main중단 실패자료는
attempt-002-disk-boundary에 보존했고 helper waiting_for_layer47/idx48 재활성화 확인.
idx51 보호는 계속 필수.
근거 model-job-c03/repair-017-retired-layer44/{AUDIT,REVIEW,CLEANUP,DISK-ADMISSION,ACTIVATION}.json.

## 최신 점검: GPU 초기화 후 Layer44 재개 / 변환 전용 화면부하 구분 적용

2026-11-06 19:07 KST module057, idx47의 Layer44를 재개했다.
19:09 확인 capture97/2309행(66.84초) 실제 전진, 가용RAM81.45GiB/디스크224.86GiB.
완료 Layer0~43/2309행/1999496토큰/전체 c03/정밀도/양자화커널은 그대로다.
이전 module056 정체 후 사용자 승인 MODE2 reset은 성공했고 작은 K4 연산
1→2→2 worker finite/exact 및 이전저장출력 exact 검사를 통과했다.
최초 GPU wedged 원인은 아직 미확정이며 특정 quant kernel 결함으로 단정하지 않는다.

GPU100 고정은 해소됐으나 실제 Chrome/KWin 화면렌더링10~11% 때문에
기존 quiet<=4 시작조건에 걸렸다. 변환 supervisor에만 별도 admission을 추가:
원래 quiet 경로 유지, 다른 경우에는 고정 boot/pdev/전체39 DRM client와 프로세스
신원을 두 coherent 표본에서 일치검사하고 검토된 ChromeGPU/KWin gfx 증분만 허용.
KFD/NPU0·GPU/JOB락, non-gfx 증분0, busy<=20 및 gfx 귀속오차<=5pp,
잘못된 카운터/변경된 client/설명되지 않는 부하는 차단한다. 벤치 quiet는 변경없음.
CPU7/infra432·독립리뷰 및 실제 gfx11.41%/2.114초 읽기전용 검사 통과.
이전464동결/STATE/실행기 보존 후 실행기만 교체하고 helper/test/contract 포함467동결.
검사 시작 가용110.9GiB/내장224.86GiB. 다른 앱 종료나 재초기화는 하지 않았다.

MTP 이전 실패 예약은 attempt-001-failed-main-stall에 보존하고 같은 helper를
명시 재활성화했다. Layer47 뒤 idx51 pre-mixer 예약 및 기존 MTP Gufo T-1 계획 유지.
helper의 알림 문구에 남은 옛 zero-first/4T 표기는 위 최신 Gufo T-1 계약으로 대체된다.
경계 reserved에서는 재개/RELEASE하지 말고 MTP 검증·보정부터 처리한다.
근거 model-job-c03/repair-016-desktop-admission/{CPU,REVIEW,ADMISSION,MIGRATION,ACTIVATION}.json,
module057/run.log 및 mtp-calibration-boundary/{REARM,BOUNDARY}.json.

## 현재 실행: module021 Layer9 / static capture 적용 완료

사용자승인대로검증후같은경계에적용하고idx12에서Layer9를재개했다.
새464동결/원본stat/2309행checkpoint/CPUadmission/디스크ready통과,
예약락released·boundary inactive·본체active·GUFO_STATIC_CAPTURE_ACTIVE확인.
가용RAM시작111GiB,디스크269.65GiB/필요192.275GiB.현재정상모듈은유지한다.
static완료callback은처리끝났으며늦은중복알림으로재시작/재검증하지않는다.
현재scope_stats의옛skipped_down=0은예상이며종료시GUFO_STATIC_CAPTURE_STATS가
새라우팅생략/count재사용횟수를기록한다.시간별감시유지.
증거repair-014-static-capture/{MIGRATION,ADMISSION,ACTIVATION}.json.

## 최신 통합 완료: static capture / idx12 Layer9 재개 준비

capture-static 검사는완료·독립리뷰통과,두candidate통합완료.기존450동결/두caller와
idx12manifest를repair-014-static-capture에보존하고새464동결을기록했다.
64행50056토큰두번516H/inf_nan/num_total/5refs/IDs/RNG 모두byteexact/반복exact,
cleanup0/KFD0.원본control재수집없음.실제capture42.43/20.75초,과거lean59.13/37.48초는
시점다른진단관측이므로동일조건전체가속률아님.합성count8192행1024회4.28초→4.2ms는
중복count연산한정이다.정밀도/전체c03/원래행렬연산/디스크사전정리/800행10GiB유지.
CPU9+runtime5/infra432(19.67초)및현재SHA검증통과.
**static 검사는끝났으므로RELEASE/락반환확인/실제CPUadmission뒤본체재개가능.**
이미재개된뒤늦은callback이오면정상본체를중단하거나검사를반복하지않는다.
근거capture-static/{STATE,SUMMARY,COUNT-ABBA,REVIEW}.json,
repair-014-static-capture/{MIGRATION,ADMISSION,ACTIVATION}.json.

## 최신 사용자 승인: 변환 중 추가 최적화 준비 / static capture 다음 경계 검사

사용자 “틈틈히 해놔야 하루 안에 끝나지”에 따라 본체와별도코드로두후보준비:
(1) pinned std all-expert의불필요router/softmax/assignment sort/bincount생략.
(2) 동일owning gather의gate/up1024회Inf/NaN검사를1회+정수배수집계로교체.
H/num_total/first_key/원래up/gate행렬연산/공유전문가tail/normalforward는유지.
정밀도·2309행1999496토큰·25%보류·현재450동결불변. 실행중파일은수정하지않는다.
CPU9그룹/runtime5그룹/infra432(19.67초)/독립리뷰통과.
gufo-static-boundary.service 및gufo-capture-static.service 대기상태확인,현재450신원전부일치.
capture_static_boundary.py가다음자연경계
GPU→JOB락을예약하고capture_static_job.py가그lease안에서최대600초검사한다.
**capture-static-boundary/BOUNDARY waiting/reserved 또는capture-static STATE 대기/실행이면
현재정상본체는유지하고경계에서중단된본체를별도재개하거나RELEASE하지않는다.**
기존64행50056토큰의저장control재사용,전체516H/inf_nan/num_total/5refs/IDs/RNG와
반복byteexact필수.합성count ABBA는해당연산만의시간이며전체가속률아니다.
실패scope미집계부분통계는진단자료일뿐재사용/승격금지.새baseline전체수집금지.
완료callback에서cleanup/실제checkpoint/독립리뷰→통과시별도static_runtime와두caller
최소통합/이전450보존/새동결/CPU검사→RELEASE/락반환확인/본체재개.
실패면자료보존후영향검사만수정하거나기존검증된450설정으로재개한다.
경계예약실패시실제락/모듈상태부터확인하며중복GPU나타작업중지금지.
후보/검사/runner:task/capture_static_*.py,capture_repeat_counts.py,
출력capture-static/{CPU,RUNTIME-CPU,READINESS-REVIEW,STATE,SUMMARY,COUNT-ABBA}.json.

## 최신 시간별 점검: Layer6 완료 / module019 Layer7 양자화 중

module018 Layer6 정상완료:2207.66초(36분48초),capture1565.66초,
quant432.72초,forward206.60초.2309행전부처리/idx10/bad_rows[]/cleanup0/KFD0.
lean실제통계:gather2309·재사용1179899·empty down/scatter생략각1182208.
본체가자동으로module019 Layer7로전진했으며capture2309행1598.89초완료,
현재양자화중이다.디스크사전검사ready/idx10,현재여유약275.6GiB·가용RAM68.2GiB.
코드/정밀도/전체calibration변경이나추가GPU검사없이정상실행유지.
Layer5/6시간차는서로다른층관측이므로최적화의동일조건가속률로단정하지않는다.

## 현재 실행: module018 Layer6 / lean capture 및 디스크 사전정리 적용

idx9/2309행에서 module018을 재개했다. 새450동결, 전체c03/FP16 Lt1·FP32 Lt3/
800행10GiB 유지. capture-lean 및 disk-boundary 완료콜백은처리끝났으므로
늦게도착해도현재정상본체를중단하거나검사를반복하지않는다.
실제사전검사 ready: free276.88GiB / 필요192.275GiB, 이경계삭제대상0.
첫시도는폐기된과거module011의idx10로그를오인해GPU시작전에멈췄다.
정리대상없는퇴역기록만skip하도록수정;미완료journal·대상있는미래idx·symlink는
계속차단. CPU23/독립재리뷰/infra432통과,기존450동결및실패기록보존후4SHA만갱신.
복구후idx9/원본신원/두락/실제diskprepare/CPUadmission통과,재계산없음.
증거 repair-013-disk-empty-history/{FAILURE,ADMISSION,ACTIVATION}.json.
실제활성확인: module018 GUFO_LEAN_CAPTURE_ACTIVE/LT lean_preparation=true,
첫60.06초capture73/2309행·scopes73.옛skipped_down_calls는새lean이대체하여0이며,
새제거횟수는종료시GUFO_LEAN_CAPTURE_STATS에기록된다.시간별감시active.

## 최신 통합: lean capture + 매 레이어 디스크 정리 / idx9 재개 준비

module017 Layer5 정상완료2507.43초, idx9/2309행/bad_rows[]/cleanup0.
capture-lean 후보2회(64행50056토큰)는516H/5정상refs/카운터/IDs/보존state/RNG
전부저장control과byteexact, 반복byteexact.59.13/37.48초는읽기/캐시조건이다른
진단시간으로전체가속률아님.peak13.41/13.57GiB.새baseline/양자화반복없음.
FP16private Gram stack 중복F32변환/마스킹과discarded std all-expert의
중복owning gather/죽은down zero mul scatter를제거.원래Lt1/3/shape/정밀도/
H누적순서/전체c03/800행10GiB/정상forward유지.25%미적용.
CPU7그룹+runtime5그룹+infra432/18.80초,독립리뷰통과.
이전433동결/두caller/idx9manifest는repair-012-lean-disk에보존했고새450동결완료.
**이제검사완료이므로disk-boundary RELEASE후락반환→CPUadmission→본체재개가능.**
이미재개됐다면늦은완료callback으로중단/중복GPU실행하지않는다.
모듈시작마다검증된과거transient만정리하고100GiB+state+16GiB필요공간을검사한다.
최종idx52의추가compile예산도적용하며공간부족은계산시작전에알린다.
증거capture-lean/{STATE,SUMMARY,REVIEW,CPU,RUNTIME-CPU}.json,
model-job-c03/repair-012-lean-disk/{MIGRATION,ADMISSION,ACTIVATION}.json.

## 최신 사용자 승인: capture 복사/폐기 출력 정리 후보

사용자 “가자”에 따라 capture_lean_gram.py / capture_lean_moe.py를 별도 준비한다.
FP16 Gram private stack의 중복 FP32 변환/마스킹 제거, 고정 std all-expert에서
첫 owning gather 재사용과 폐기되는 routed down zero/mul/scatter 제거가 대상이다.
원래 Lt 알고리즘/정밀도/H 누적순서와 normal forward는 유지한다. 아직 미적용.
module017 Layer5 정상완료 idx9에서 disk-boundary 예약이 두 락을 보유중이다.
**capture-lean 준비/검사/리뷰가 끝나기 전 RELEASE나 본체 재개 금지.**
CPU/독립리뷰 후 기존64행/50056토큰 control 재사용으로516H/5refs/IDs/RNG exact를
묶어 검증한다. 실패는 해당 검사만 수정, 새 baseline 전체수집/25% 적용 금지.
같은 경계에서 검증된 디스크 사전정리도 통합하고 체크포인트에서 재개한다.

## 최신 사용자 요청: 매 레이어 시작 전 정리 / 자연경계 적용 예약

사용자 “레이어 시작하기 전에 안 쓰는 거 지우고 시작” 승인에 따라
`disk_boundary.py`와`model_job_disk_candidate.py` 준비, CPU20/infra432(19.58초)/
독립리뷰 qualified_for_boundary_housekeeping. 현재 본체433동결은 그대로다.
`gufo-disk-boundary.service`가자연경계GPU→JOB락을대기하고완료callback을보낸다.
**disk-boundary/BOUNDARY가 waiting/reserved이면 현재 정상모듈은유지하되,
경계에서 멈춘 본체를 별도로 재개하지 말고 아래 통합을 먼저 처리한다.**

정리 정책: 검증된2309행·token IDs 길이·현재inode/미완료journal 검사를 먼저 한다.
성공/복구commit확정된과거attempt의row-scratch/previous-checkpoint/partial-checkpoint만
정리하고목록을사전fsync한다.실패미복구/실행중자료와모델·baseline은보호한다.
매레이어시작공간은100GiB reserve+현재state실측+16GiB(현재192.275GiB).
마지막idx52는qtensors전체실측+20GiB를추가(새head/MTP/vision작업·출력복사상한),
완료ngram48GiB는재복사하지않음.부족시에만unused Docker buildcache>24h 정리;
그래도부족하면GPU계산을시작하기전에receipt+동일채팅알림을남겨공간확보후재개.

경계callback: 이전module성공/실제checkpoint/cleanup확인→이전433 FROZEN과
model_job.py/STATE보존→검토된model_job_disk_candidate.py만caller교체→새437동결
(추가helper/test/CPU/REVIEW 4개)→RELEASE작성/예약락반환확인→CPUadmission+
prepare_disk의실제사전정리/공간검사→기존본체unit재개→첫disk-boundaries영수증확인.
후보·원본SHA를disk-boundary/{CPU,REVIEW,INTEGRATION-CPU}.json과대조한다.
실패module이면먼저복구하며임의로통과상태로바꾸지않는다.실행중코드수정금지.


## 최신 디스크 정리: 미사용 checkpoint 잔여84.24GiB 회수

사용자가미사용checkpoint삭제를명시승인.독립inode/참조감사후
module014/row-scratch(중단한idx6입력)와repair-002-checkpoint/truncated-ckpt-new/
state.safetensors(잘린무효9GiB파일)만삭제했다.작업STATE/log/job/잘린header/stat보존.
2293개단독inode회수,공유16개행은capture-io/fixture에남아inode/size재검증통과.
현재idx8/실행module017과겹침0,idx2진단입력·완료packed·baseline보존.
free197.43→281.67GiB,본체는중단하지않음.근거repair-011-unused-checkpoints/CLEANUP.json.
다음시간별점검에서는새state약76.275GiB+새packed+100GiB reserve를고려한다.
시작여유185GiB이상원칙유지,현재281GiB가전체완료까지무조건충분하다는뜻은아니다.
최종compile의추가출력공간도전단계에서확인하며,부족예상시재생성캐시/명확한미사용
중복자료를미리정리한다.활성checkpoint나실행scratch는지우지않는다.


## 최신 실행: idx8/Layer5 재개, pair 후보는 미적용

Layer4 disk중단은2309 완료행 CPU복구/독립리뷰로idx8을확정했고,433동결/원본신원/
GPUquiet/가용111GiB/디스크197.43GiB(시작185GiB조건)를검사한후module017 Layer5재개.
800행10GiB읽기큐·전체c03·원래FP16 Lt Gram 유지, 추가pair 후보는미적용.
repair-010-disk-checkpoint/{ADMISSION,ACTIVATION}.json 참조.

pair 실제검사는133.30초에완료,cleanup0/KFD0.64행/50056토큰에서516H카운터/
5정상참조/IDs/RNG일치,512expert down H만변경(maxrel1.260194e-5),나머지4H byteexact,
후보2회전체numeric반복byteexact.시간64.07/43.06초·weightpack0.162초,
peak약23.24GiB.작은연산20%이득의실제속도재현은미확정(읽기/캐시조건다름),
추가메모리약8GiB.양자화품질은아직검증하지않았다.새전체baseline/양자화검사없음.
후보는저장자료를보존한미적용상태이며구조/메모리개선후필요한검사만이어간다.
동일완료콜백이늦게도착해도현재main을중단하거나중복검사하지않는다.


## 최신 복구: Layer4 완료행으로 idx8 확정 / pair 검사 준비

module016은 Layer4 양자화·전체forward를2528.59초에 완료했지만 저장직전
디스크100GiB guard에 걸려 중단됐다. GPUcleanup0/KFD0. pair 첫 시도는 이전모듈
성공 gate에서 차단되어 **GPU 실행 자체가 없었다**. attempt-001-boundary-disk 보존.

recover_disk_checkpoint.py 별도 CPU복구는 frozen finish_checkpoint의 모든 검사를
유지하고 정확한 disk-reserve 오류+완료forward만 추가허용했다. CPU9/독립리뷰,
2309 새출력(76.275GiB)/1542 packed/IDs/mtime/extent/완료로그 통과 후
1.68초에 idx8 hardlink commit. journal committed/모든inode 검증 후 일반 checkpoint
rotation으로 이전idx7/부분링크/scratch를 정리했다. 완료된 GPU계산은 반복하지 않았다.
오래된 미사용 Docker buildcache28.85GB 정리, 회수후 free200.92GiB.
원본실패STATE·양자화pack·새idx8·기존baseline은 보존.
근거 module016/{DISK-RECOVERY,CPU-COMMIT}.json, repair-010-disk-checkpoint/.

**현재 본체는idx8에서대기, pair 검사를 먼저 처리한다.** pair runner는 독립 검토된
DISK-RECOVERY와실제idx8을검사한경우에만이복구경계를수용한다. 검사 결과 리뷰 뒤
기존433동결/800행10GiB로재개한다. 다음 GPU본체 시작 전185GiB 이상 확보:
기존 admission150GiB와실행reserve100GiB 사이50GiB는새출력76.275GiB를담지못한다.
현재 본체 frozen guard는 바꾸지 않았다. 시간별점검은 transient출력과 실제free를
구분하고, 다음모듈 공간부족이예상되면사용하지않는재생성캐시를먼저정리한다.


## 최신 후속 준비: 전문가 gate/up Lt 실제 capture 검사

사용자 “코딩이라도 미리” 지시에 따라 실행 중 본체와 별도 파일로
capture_forward_pair.py/probe/job을 준비했다. 본체433동결은 변경하지 않는다.
GPU를 사용하지 않는 CPU 검사와 독립리뷰 뒤 gufo-forward-pair.service가 다음
자연 모듈경계의 globalGPU→JOB락을 예약한다. **capture-forward-pair/STATE가
waiting_natural_boundary/running_capture_screen이면 별도 본체 재개/중복GPU 금지.**
현재 정상 module016은 중단하지 않으며 경계에서 supervisor가 LOCK_NB 실패로
끝나는 것은 예상 인계다. 실제 이전 module 성공/checkpoint/cleanup을 먼저 확인한다.

검사는 기존 capture-pipeline의 layer6/idx9 64행/50056토큰과
capture-lt-evaluate/1-original_f16/numeric.pt 성공 control(SHA INPUT.json)을 재사용.
기존 baseline 재수집 없이 후보 두 번:516 H/카운터/5정상참조/IDs/RNG/반복byte일치,
expert down 이외 H 불변을 확인한다. 모든 양자화/모델출력/체크포인트쓰기 금지.
원래expert FP16입출력, Lt0 B2 up/gate만 변경하고 FP16 Lt Gram은 그대로 사용한다.
outer Linear.capture_H를 보존하며 discarded scope만 변경, 정상 reference는 원래 경로.
읽기전용 row loader가 기존과 달라 시간은 진단용이며 전체/실제속도 개선 주장은 금지.

up 출력은 down H가 참조하므로 전문가별 새 B2출력을 사용한다. 메모리추가량은
packed weights3.125GiB 외에8K행에서소비된 gate storage 약5GiB가 flush까지 남을 수
있다. 4GiB는 weight pack 상한일 뿐 총메모리 상한이 아니다. plan8/256MiB workspace,
기존host8GiB/disk100GiB 실행guard·600초창·readonly원본·소유컨테이너cleanup 유지.
검사 종료 callback 후 독립리뷰: H변화만으로 품질동등이라 하지 않고 유망후보만
양자화/출력검사로 진행, 생산 자동적용 금지. 실패시 저장부분H/numeric 재사용.
최신 사용자 지시 우선; 25%미적용/800행10GiB 유지. 본체는 검토 후 기존승인으로 재개.


## 최신 상태: 800행/10GiB 통합 완료, idx7에서 재개

module015/layer3 정상 완료: 총2600.42초, capture1938.16초, quant430.94초.
checkpoint idx7/bad_rows[]/2309행/1999496토큰 확인, cleanup0/KFD0.
이전428동결/두caller/상태는 repair-009-large-prefetch에 보존했다.
검토된 large 후보 caller 두 개만 설치하고5의존 추가, 새433동결 전부 일치.
독립리뷰→RELEASE→boundary inactive/두락 반환→실제 supervisor CPU admission
및 원본신원 검사 통과→기존 본체unit 재개. module016/layer4가 다음 작업이다.
읽기800행/10GiB 중 먼저 도달하는 상한, 쓰기2GiB; 현재행/중간tensor는 별도다.
25%미적용, 전체c03/기존 FP16 Lt Gram64/owner수정 유지. 시간별감시 active.
근거 repair-009-large-prefetch/{REVIEW,ADMISSION,ACTIVATION}.json.

같은 경계의 capture-forward 연산검사는 완료·cleanup 정상이며 REVIEW에 실험 유지.
expert gate/up B2 Lt0 약20%단축, dense Lt4 약4.56배지만 FP64표본 오차 증가.
현재 본체에 forward 후보 hook은 없다. 추가 최적화는 실제capture/H/카운터/정상참조/
양자화검증을 통과해야 한다. 저장 input/control을 재사용하고 현재 실행 중 코드는 수정하지 않는다.
WMMA 작은 수치차이는 미검증이며 품질하락 확정이 아니다. 별도 small-M 원본hgemm
CPL/grid 결함은 c03 최소512행에는 해당하지 않는다. 상세 capture-forward/REVIEW.json 및
GFX1151-KNOWHOW.md. 이전 '검사대기 중 release금지' 조건은 이 완료검사에 대해 해제됐다.


## 최신 사용자 추가: 통계수집 나머지 Lt/WMMA 최적화

사용자가나머지forward/전처리까지Lt·WMMA최적화를승인.현재layer3는유지한다.
**large-prefetch 경계콜백은capture-forward/STATE.json부터확인: 검사대기/실행중이면
RELEASE.json작성이나본체재개/검사코드수정금지.같은경계에서연산검사를묶는다.**
`gufo-capture-forward.service`는기존large boundary GPU/JOB락의PID/dev/inode를
검증하고이전모듈성공/cleanup뒤최대600초검사한다.자기OWNER락/readonly원본/
RAM8GiB·disk100GiB/자기컨테이너cleanup/중지핸들러,완료같은채팅콜백.

후보capture_forward.hip/ops/probe:직접Lt(최대8알고리즘),같은입력을공유하는
expert gate/up B2묶음,WMMA.원래expert gate/up출력은FP16,dense출력은FP32임을
실제Config CPU검사로확인했다.초기expertFP32가정은GPU실행전검사에서실패하고수정,
attempt-001-cpu-contract에보존.원래FP16입력/FP32누적/실제출력형식유지.
CPU6/커널빌드/독립review통과,actualGPU미검증.초기FP32전문가라는설명은정정됨.

작은정수tail2dtype+실제layer0/보존idx2의512·8192행에서gate/up/dense입력만관측,
전체H나기존baseline재수집안함.새연산control/입력/출력/FP64표본/반복/ABBA보존.
weights hot-cache·원본출력할당/후보버퍼재사용차이가있어순수GEMM/전체속도주장금지.
검사후유망후보만실제capture/H/normalrefs/count/최종quality검증으로확대한다.
결과capture-forward/{STATE,SUMMARY,PROGRESS,ANALYTIC,BUILD,CPU}.json.
느림/불일치면저장자료로해당작은검사만수정,본체동결자동교체안함.

후속우선순위:1)expert gate/up묶음/Lt 2)dense FP32출력Lt
3)원래FP16 H입력의불필요FP32복사·중복mask제거(현재halfLt도xf를만들지만안씀)
4)discarded all-expert capture의불필요gather/zero-scatter/반복finite처리.
3·4는아직후보미구현,기존H누적/카운터/원본normalrefs보존을검증해야함.
전체성능/품질검증은변환후묶고,800행10GiB설정도다음재개전에통합한다.

## 최신 요청: 읽기큐 최대800행/10GiB 확대 준비 / 현재모듈 유지

사용자가8행대신800행,읽기상한10GB와더많은RAM활용요청.
`disk_state_large.py`는검증된AsyncRows를상속해read10GiB/ahead800만변경,
write2GiB/현재100GiB컨테이너/8GiBhost reserve/GPU산술불변.
두상한중먼저도달:실제첫window319행10602127360bytes,512토큰행512개=10GiB.
CPU기존9계약+정책3검사통과.전체baseline/GPU반복없음.
`convert_main_large_candidate.py`/`model_job_large_candidate.py`준비,현재428동결불변.

`state_large_boundary.py`가자연모듈경계에서globalGPU→JOB락을확보하고
capture-io-large/BOUNDARY.json reserved와같은채팅후속을전송한다.현재모듈은중단안함.
예약중본체supervisor는LOCK_NB획득실패로정상경계에서종료할수있음:
이경우모듈성공/checkpoint진전부터확인하고계산실패로오분류하지않는다.
다음통합:독립REVIEW/CPU확인→이전428동결/두caller/STATE보존→두후보교체→
새경로선언FROZEN(large_root필요)생성→CPUadmission→RELEASE.json작성→
예약프로세스종료/락반환확인→기존본체unit재개→10GiB/800활성로그확인.
실패모듈이면원인검토우선,최신중지지시우선.실행중코드수정/새GPU검사금지.
근거capture-io-large/{CPU,INTEGRATION-CPU,REVIEW,BOUNDARY}.json.

## 현재 실행: 비동기 I/O 적용 / layer3 재개

사용자명시지시로module014중단후idx6보존,읽기worker1/쓰기worker1의
bounded비동기I/O를적용했다.disk_state_async.py:최대8행/읽기2GiB/쓰기2GiB,
CPU snapshot소유·read-after-write/오래된prefetch무효화·checkpoint drain/error전파.
현재행/GPU결과/clone중간값은큐상한별도.100GiB컨테이너/8GiBhost reserve유지.
GPU계산/H누적순서/전체c03/Lt64/owner수정불변,25%미적용.

CPU9검사·실제idx6첫16행GPU전송/elementwise/I/O ABBA byteexact4/4,
원본manifest/IDs/2309행stat보존·cleanup0/KFD0.시간.83949/.74239/.72310/.72396초:
warm원본대비동일/노이즈,전체모델속도개선주장없음.전체baseline반복없음.
독립리뷰qualified_for_io_integration/infra432(18.95초run-02c3k3fw)/CPU admission428통과.
기존419동결보존후두caller만변경+9의존추가.근거capture-io/{STATE,RESULTS,CPU,REVIEW}.json,
model-job-c03/repair-008-state-io/{MIGRATION,ADMISSION,ACTIVATION}.json.
본체module015/layer3재개.실행중코드수정금지;시간별감시유지.
완료한one-off state_io_job.py는SIGTERM handler한계기록,재사용금지.
본체는기존SIGTERM cleanup/체크포인트보호경로유지.최종전체검증은변환뒤묶어서수행.

## 현재 상태: prefix 복구 완료 / owner 수정 통합 후 layer3 재개

전체 prefix 재생은1813.25초에정상완료.2309행/1999496토큰/idx6,
첫3행pilot byteexact·모든행finite·원본보호·cleanup0/KFD0.정상0/PLE/1/2
packed만재사용했고새양자화없음.독립리뷰후`prefix-recovery/REVIEW.json`에
qualified_for_idx6_promotion기록,동일파일시스템rename으로`work/ckpt`승격.
`PROMOTION.json` committed_idx6:2309행inode/header/extent보존,IDs/manifest SHA기록.

항상실행q_proj가공유H를소유하도록capture_owner최소통합.기존405동결중
convert_main.py/model_job.py만변경,의존14추가로419동결.원래FP16 Lt64/전체c03/
정밀도/worker2유지,25%미적용.기존CPU7검사와동일후보를설치했고실제supervisor
CPU admission/전체419신원/원본inventory검사통과.최종독립리뷰/infra432개18.99초
(run-v7qq68_y)통과후기존unit시작,module014/layer3재개.시간별timer active.
근거repair-007-capture-owner/
{MIGRATION,ADMISSION}.json.본체layer3(idx6)재개후ACTIVATION.json확인.

사용자는작업에약90GiB를유용하게활용해속도를높이길원함.현재Docker상한100GiB,
기존안전reserve유지.읽기prefetch/GPU/비동기쓰기파이프라인은미적용후보이며
이번owner복구와섞지않음.실행중파일수정금지,전체모델검증은변환후묶어서진행.

## 이전: 전체 prefix 복구 / layer3 재개 대기

`prefix-recovery/pilot/` 45.74초완료·3행idx6·IDs/finite/원본보호/cleanup0/KFD0,
독립REVIEW `qualified_for_full_prefix_replay`. 첫시도는GPUforward전표시용weights_numel
미로드조회에서실패했고attempt-001에보존;표시계산제거후작은검사만재시도했다.
CPU계약6/owner통합7·infra432(18.88초run-395pzcm7)통과.
**gufo-quant-prefix-replay.service full실행중**. 전체2309행/1999496토큰,0/PLE/1/2
기존packed순전파만실행,원래module별checkpoint로idx6까지복구한다.
진행/실패시 `prefix-recovery/STATE.json`, `run.log`, `work/ckpt/job.json` 확인.
실행중코드수정/중복GPU금지. 정상인full은그대로유지.

완료콜백에서cleanup·전체IDs/manifest·첫3행pilot exact·원본보호/독립리뷰→
`prefix-recovery/REVIEW.json`status=qualified_for_idx6_promotion→복구ckpt를현재비어있는
`work/ckpt`에rename승격. `PROMOTION.json`status=committed_idx6,manifest_sha256/ids_sha256/
행inode매칭을기록. old405FROZEN/실행기는repair-007에이미보존돼있다.
준비된 `convert_main_owner_candidate.py`/`model_job_owner_candidate.py`만최소교체하고
경로선언으로새FROZEN생성/CPUadmission후기존gufo-quant-model-c03.service start.
model_job후보는ownerAUDIT해시·prefix CPU/3코드/승격영수증·첫3행결과를연결한다.
추가새baseline/건강한0~2재양자화금지,검증완료본체재개는사용자명시승인범위다.

사용자질문확인: layer3당시countguard가없었고오류없이완료처리됐다. Lt통합때guard를
추가한뒤layer7에서처음누락발견.0~2는조건부indexer경로없고당시nonfinite/fallback없음.
이는이번결함의미영향근거이며0~2전체품질전수검증완료주장이아니다.

## 최신 사용자 승인: 무효 결과 삭제 / layer3부터 복구 실행

사용자가“기존거못쓰는거면유지하지마그냥지워버려”,“정리하고3부터진행하자”승인.
이전idx10/불완전layer3~6packed보존지시를해당자료에한해대체한다.
`repair-007-capture-owner/DELETION.json`에정확한삭제목록/메타보존. idx10+failed012scratch+
layer3~6packed삭제81.29GiB회수,미사용Dockerbuildcache42.11GB정리. 정상idx2/0~2/PLE/K8ngram보존.

**현재work/ckpt는의도적으로없으며본체를시작하면안된다.**
`prefix-recovery/`에서기존idx2입력과packed modules2..5(layer0/PLE/layer1/layer2)만
사용해원래converter stateadvance/checkpoint경로로idx6복구한다. 캡처/양자화/compile금지.
`prefix_replay.py`는pinned원본main의capture/quant영역만packedload로교체하고나머지순전파/
quant_preserve/finite검사/행단위checkpoint는원본그대로다. module별저장으로재개가능.
첫3행pilot `gufo-quant-prefix-pilot` 뒤2309행full `gufo-quant-prefix-replay`.
pilot600초/full7200초,globalGPU+JOB락,readonly원본,RAM28/8·disk150/100GiB,
원본stat/packedSHA·정상first3pilot↔full byteexact검사. 같은검사/양자화baseline반복없음.
CPU AST계약6검사/독립리뷰; 실행자료 `prefix-recovery/{CPU,INPUTS}.json`.
full종료시같은채팅에즉시검토/idx6승격/owner통합/새동결/본체재개후속을보낸다.
생산후보 `convert_main_owner_candidate.py`, `model_job_owner_candidate.py` 준비됨(아직미적용).
각unit실행중에는중복GPU/코드수정금지. checkpoint진행여부는prefix-recovery/work/ckpt를읽는다.

## 최신 진단 완료: capture owner 수정 검증 / layer3 복구 필요

`capture-owner/REVIEW.json`: 실제layer7의3217→512행에서기존count3217→3217로누락재현.
항상실행q_proj owner후보는3217→3729이며,유효한기존short-first와516개H모두byteexact,
모든count3729·normalrefs/Q입력exact. FP64표본오차7.83e-6. CPU7/infra432통과·독립리뷰.
GPU검사의추가조건'broken대비other515H exact'는mlp.input relL2 8.34e-8로실패했으나,
저장된정상short-first oracle와전체일치를CPU재검토확인했다. 실패STATE보존/GPU재실행없음.
cleanup0/KFD0. 후보코드는 `capture_owner.py`; 생산동결405개/실행기미변경.

현재본체는idx10중지. layer3의indexer/Q/K/V도동일owner이력영향대상이며그뒤4~6입력도
바뀔수있으므로idx10단순재개는올바른전체2M수정본이아니다. 다음최소복구:
보존repair-002/checkpoint-idx2 + 기존qtensors layer0→layer1.ple→layer1→layer2로
**양자화재실행없이state만재생**하여idx6(layer3입력)복원,그뒤layer3부터수정적용.
기존idx10/완료packed/idx2원본삭제금지. 생산통합과prefix복구는아직시작하지않음.
다음점검은중복본체를띄우지말고이복구준비/리뷰/자원확인을이어간다.

## 최신 중단: module013 공유 attention H 담당 선택 결함 조사

module013(layer7)는전체capture1969.08초완료뒤count guard가indexer에서실패했다.
cleanup0/KFD0,checkpointidx10보존. **자동재개금지: 실패한검사를약화하지말고
capture-owner 소형재현/수정과기존layer3영향범위검토뒤명시복구한다.**
정적원인:sharedblock.attn.input의first_key가첫3217토큰행에서조건부indexer로설정되고,
seqlen<=2051인2151개512행에서는indexer미호출로H수집누락. 예측count898184대신
전체1999496이어야함. Lt산술과무관한기존capture-owner문제이며실제작은재현진행중.
기존module008(layer3)도같은행순서이므로검토대상.완료packed/idx10삭제금지.

## 최신 실행: Lt64 통합 후 idx10 재개

사용자추가제안Lt128검사완료: `capture-lt-chunk/` 64→128→128→64 실제루프는
65.7576/44.7009/44.6832/44.6663초. 첫64의cold/order차이를128개선으로계산하지않는다.
마지막64와128은동일수준이며128은Torch peak+3.865GiB여서**64유지**.
모든case가저장Lt64 H/ref/count/IDs/RNG byteexact,추가304shape지원,cleanup0/KFD0.
기존baseline/품질검사는재실행하지않았다. 독립리뷰 `capture-lt-chunk/REVIEW.json`.

`capture-lt-support/`1520shape지원+attention33/512/8192후보2회정상,
`REVIEW.json`·runtimeCPU7·infra432 통과. 원래FP16입력만Lt1/F32누적,
FP32입력은Lt3 F32유지. **chunk64·8plan/256MiB·미지원명시실패·silent fallback없음**.

`model-job-c03/repair-006-capture-lt/`에oldFROZEN393/기존2소스/STATE/idx10metadata보존.
기존2실행기만변경+12의존추가=새FROZEN405,CPU admission검사통과.
모든2309행checkpoint header/extent/shape정상,1999496tokens/c03/recipe/기존완료packed유지.
76GiB checkpoint별도복제없이기존rolling저장방식을유지한다.25%후보계속보류.
**gufo-quant-model-c03.service attempt013(layer7)실행중. GUFO_LT_ACTIVE에서
원래FP16입력/F32누적/Lt1 활성확인. 기록 repair-006-capture-lt/ACTIVATION.json.**
정상실행중코드수정/중복GPU금지. 최종전체모델정확성·성능은변환완료후한번에검증.

## 최신 사용자 승인: FP16 Lt 적용 후 idx10 재개

사용자가“가자가자~”로남은attention/shape확인→FP16Lt통합→idx10재개를승인했다.
이전중지지시를이순서의작업에한해해제한다.25%는계속보류/전체200만토큰유지.
`capture-lt-support/`먼저확인:실제151길이+검사33의1520shape/precision heuristic생성확인,
저장attention baseline을재계산하지않고33/512/8192후보2회검사한다.
GPU unit `gufo-quant-capture-lt-support`가실행중이면본체/다른GPU를중복시작하지않는다.
통과·cleanup·독립리뷰뒤이전동결/코드/idx10metadata보존,최소통합/CPU검사/새동결로재개한다.
실제FP16입력만Lt1/F32누적,FP32입력은Lt3 F32유지.8plan/256MiB상한,미지원명시실패.
완료checkpoint/기존packed/원본baseline은재계산하지않고최종전체검증은변환완성후묶는다.


## 최신 검토 완료: 원래FP16 Lt 우선후보 / 본체 미적용·중지 유지

`capture-lt-quality/`177.35초완료cleanup0/KFD0,독립리뷰완료. 실제BF16source teacher 대비
동일64행H양자화의8문서평균오차비:GDNout FP32 0.99744961(-0.2550%),
원래FP16 Lt 1.00001649(+0.00165%). expert0는두후보모두packed전체byte와출력오차동일.
FP16 GDN의최악문서오차비1.006148(+0.615%),최악문서sampledp99비1.018822(+1.88%).
모든지표동등/개선주장아님.6quant모두CPU FP32 Cholesky첫시도성공/rotation동일/fallback0.
BF16→loaderFP16 값차max2.98e-8은별도기록,teacher는실제BF16을사용했다.
CPU FP64표본교차검증보존;작은표본의오차비를전체모델성능으로확대하지않는다.

현재추천은**원래FP16입력에Lt1·원래FP32입력에Lt3 유지**후보다. FP32입력하향변환없음.
실제capture후보간53.52→44.85초(16.2%단축),입력/count/IDs/RNG/refs/후보반복정상.
기존baseline재수집0·원본비교시간은historical. 최초별도64행불일치원인은미해결로보존.
범위는layer6 GDN두projection/고정8문서이며전체품질/KLD검증아님.

다음통합전필수:full-attention·남은shape/dtype의선택Lt algorithm지원확인,
미지원조합정책명시(실패후silent fallback없음),bounded8plan/256MiB와무결성검사유지.
**현재추가GPU검사는종료했고생산통합/본체재개는하지않았다. idx10·25%보류유지.**
H/8문서layer6입력/prefix중간상태/teacher/packed/실패자료를재사용한다.
증거:`capture-lt-quality/{SUMMARY,QUALITY-AUDIT,PACKED-AUDIT,REVIEW}.json`.

## 후보실제루프 완료 / layer6 원본BF16 대비양자화오차 검사

`capture-lt-evaluate/`381.90초완료cleanup0/KFD0. warm/observer제외FP32평균53.52초,
원래FP16입력후보44.85초:두후보사이16.2%단축.6case모두normal refs savedcontrol exact,
반복후보H/ref exact·참조사후보존exact.12evictions/peak8plan/256MiB상한정상.
과거원본54.9~55.0초는historical참조이며동시원본대비개선인증아님. 최초불일치원인미해결.

`capture-lt-quality/`에서는저장원본/새후보64행H(50056tokens)로layer6GDNoutK6/expert0downK4를
동일RNG/worker로양자화한다.이전에없던같은layer6/같은H표본의projection reference만계산,
전체원본baseline/H수집/이미완료packed는재실행하지않는다. 고정8문서IDs재사용·modules0..8
원본loader+고정K8ngram전진후입력저장. teacher는실제BF16sourceweight의FP32matmul이며
CPU FP64 16행×64출력열표본교차검사. BF16↔loaded값관계도기록한다.
rotation/finite/fallback0/Cholesky검사;추가damping시원본실패자료보존후실격.
이전advisory문서오차비mean1.02/max1.05유지.50056tokens는fullrank보장아님.
prefix중간상태/H/teacher/packed보존,600초상한·직렬GPU·CPU5/infra432·독립리뷰.
두projection고정8문서평가이며전체모델품질/KLD증명이아니다. 본체idx10/25%보류유지.

## 최신 승인: byte 일치와 품질을 구분해 실제 후보 평가 진행

사용자는원본보다정확한가/실제로쓸만한가검토를“진행해”로승인했다.
`capture-lt-evaluate/`는저장원본control/시간을재사용하며원본capture는재실행하지않는다.
FP32observer64행(첫5참조즉시/끝보존),FP16warmup후FP32/FP16/FP16/FP32후보끼리측정한다.
observer/warmup은시간집계제외. 입력/IDs/preserve/count/shape/finite와참조사후보존은엄격검사,
H·생성출력·후보반복수치차이는품질평가자료로기록하며차이자체를품질실패로단정하지않는다.
최초exact실패영수증은불변이며,이평가로실패를통과로바꾸지않는다.
600초/직렬GPU/readonly보호/8plan256MiB/ownedcleanup. CPU13검사/infra432·독립리뷰.
기존54.9~55.0초는historical참조로만사용하고동시matched속도검증이라고주장하지않는다.
유망하면저장원본64행H·새H로같은layer6 projection의양자화오차평가를이어간다.
본체idx10중지/25%보류/생산미연결유지. 이진단승인은본체재개승인이아니다.

## 최신 결론: bounded Lt 미통합 / 최초64행 실패 원인 미해결

`capture-lt-prefix/`60.70초완료·cleanup0/KFD0. 첫5참조생성직후와5행끝모두
저장control byteexact,516H count19848·입력/preserve/IDs/RNG보존.
실제LRU eviction12회/peak8plan/명시적workspace256MiB상한확인.
Torch peak15.20GiB는Lt workspace·라이브러리내부메모리를포함하지않는다.
이는5행정확성/수명검사이며,60.70초는준비·해시·CPU비교를포함해성능결과가아니다.

최초`capture-lt-pipeline/`64행불일치의원인은아직미해결이다. 원본선실행과cold Lt-first
row0/5행에서는재현되지않았지만무해/품질동등/수정완료로판정하지않는다.
A/B/C/C/B/A는시작하지못했으므로후보채택/전체속도결론은보류.
FP32Lt3은기존연산/flush/작은양자화증거상우선후보로남기고FP16Lt1도별도후보유지.
후속범위:원본baseline재실행없이후보64행의첫5참조즉시복사와끝시점참조를함께보존해
나중변조/프로세스간재현성을분리한다. **이후검사는아직시작하지않았다.**
본체idx10·FROZEN393파일불변,25%보류,생산변경/재개0,진단GPU모두반환.
변환은완료되지않았으며시간별감시의과거재개지시로자동재개하지않는다.

## cold Lt-first row0도 exact / 첫5참조와 eviction만 검사

`capture-lt-cold/`21.05초완료·cleanup0/KFD0:첫실제cm참조와추가4회정상출력
모두저장control byteexact. 원본선실행필요가설도지지되지않는다. 최초64행불일치는미해결.
`capture-lt-prefix/`는처음5행만원래capture를실행하고참조를생성직후CPU복사해
5행끝에서보존여부를확인한다. 새GPU연산/중간GPU sync/추가normal은넣지않는다.
516H count19848,IDs/RNG/입력/preserve,실제plan eviction/상한검사.300초상한.
전체64baseline재실행0,통합/본체재개0,25%보류. 통과만으로최초실패해결선언하지않는다.

## Lt 원인분리: 원본 선실행 시 exact / cold Lt-first row0 검사

`capture-lt-ref/`32.00초 완료·cleanup0/KFD0:8회정상forward+실제cm참조가
모두저장control byteexact. 원본flush/Ltflush/planclose전후변화0,입력/pending/ID/RNG보존.
4plan peak/eviction0이라8plan eviction검증은아님. 최초실패원인해결아님.
다음`capture-lt-cold/`는실패순서대로첫Ltflush→cm참조0저장후에만normal반복/close검사.
원래host progress·entry sync도맞춤. 첫행만/300초/본체중지/25%보류·생산미연결.

## bounded Lt 실제루프 실패 / row0 원인분리

`capture-lt-pipeline/` 첫FP32Lt warmup72.64초 후 정상참조0 byte불일치로중단.
저장실패numeric CPU분석:5참조 finite·상대L2 7.85e-7~8.42e-6,
516H metadata/count·preserves·IDs·RNG·input hashes일치. H변화는별도이며
정상forward에H가전달되는코드경로는없다. 원인미확정/속도비교미완료/후보미적용.
cleanup0/KFD0, 본체idx10중지/25%보류. 원본baseline/실패자료보존.
`capture-lt-ref/`는첫행만실행:동일normal반복→원본flush→반복→Ltflush→반복→close후반복.
입력/preserve/IDs/RNG/pending불변확인,진단H는두번누적하나row1/양자화전에반드시종료.
CPU5계약·infra432/독립리뷰,300초상한. 전체64행baseline은다시실행하지않는다.

## bounded Lt 실제64행capture 루프 검사 준비

`capture-lt-pipeline/`: Lt plan LRU8로명시적workspace상한256MiB(별도라이브러리내부메모리제외),
eviction전같은stream완료후해제·새할당. CPU100shape수명/상한9개·산술20·observer/numeric6검사,
infra432 및독립리뷰. 이전unboundedhelper는capture-lt/source에보존한다.
실제cm.main64행fixture와저장control로원본/FP32Lt3/원래FP16Lt1을A/B/C/C/B/A비교한다.
원본전체strict exact,후보H차이는기록·나머지count/refs/IDs/RNG strict exact,후보반복byteexact.
매case빈plan cache로시작하며생성/eviction동기화비용을포함한다. peakTorch+cache최대+hostreserve기록.
최대600초,원본readonly/fixture별도복사·GPU직렬·ownedcleanup. 결과후독립리뷰하며
본체idx10중지/25%보류/생산미연결은유지한다. 실패partial H/numeric 보존·기준재사용.

## 실제flush/같은표본 양자화 완료 / bounded cache 후 실제루프 검증 필요

`capture-lt/` 완료37.37초,cleanup0/KFD0·생산FROZEN불변·독립리뷰완료.
원본flush평균0.73544초,FP32Lt0.36042초(2.04배),원래FP16Lt0.12585초(5.84배).
원본은저장control exact,후보반복byte안정/카운터정확. 동일H표본양자화fallback0/rotation동일.
8문서문서별평균출력오차상대증가:dense FP32+0.0435%/FP16+0.1032%,
expert FP32+0.0154%/FP16+0.0140%;최악sampledp99증가약1.92%.
두연산/작은H표본검사로전체품질동등아님. dense512는damping전rank부족제한.
다음후보선별은실제64행capture루프의refs/IDs/count/누적H/시간/메모리검사.
**현재LtGram.plans는shape마다32MiB를무제한보유하므로생산전bounded cache 또는
shared serial workspace/회수정책필수.** 이를보완하고작은검사/리뷰뒤진행한다.
원본200만토큰유지,25%보류,본체idx10중지,아직생산연결없음. 증거REVIEW.json보존.

## 실제flush 및 같은표본 양자화 소형 검사 준비

`capture-lt/`: 저장real-inputs/control재사용,원본flush의BMM한문장만직접Lt로교체해검사한다.
후보는FP32 Lt heuristic3과원래FP16입력 Lt heuristic1. 64group/유한값mask/순차Hadd/count유지,
FP32입력은낮추지않으며미지원은실패로기록한다. 새생산설정/25%적용은없다.
expert64×(512+8192)와dense1×512를쌓기/형변환/mask/H누적포함두ABBA로검사한다.
원본은저장Gramcontrol과byteexact필수,후보는반복byte안정성/실제카운터/수치차이를기록한다.
동일작은H표본·가중치·RNG·workerstream으로GDNoutK6/expert0downK4를원본/FP32Lt/FP16Lt
각각양자화하고기존8개문서입력에서출력오차/p99비교. fallback금지/rotation동일필수.
dense512H는damping전rank부족이라모델전체품질근거가아니다. H/평가입력/packed는보존한다.
CPU20/infra432·독립리뷰후최대600초격리실행. 본체idx10중지·25%보류유지.

## Gram 후보 연산 검사 완료 / 다음은 실제flush 및 같은표본 양자화 비교

`capture-gram/` 완료97.35초·cleanup0/KFD0·원본/FROZEN불변·독립리뷰완료.
FP32 Lt2/3은512expert2.1~2.2배,dense3.2~3.5배,8Kexpert2.4~3.7배.
FP64표본오차는expert512/8K에서감소,dense약1.35%상대증가. 기존과byte동일은아니다.
8K FP32Lt1은byteexact·약7%개선. 원래FP16 Lt는약7.2/9.9/17.5배이나8K표본오차가
기존약9.6e-7→1.7e-5로늘어별도양자화품질검사가필요하다. customWMMA는3.2/6.0/6.1배,
Lt보다느리고동일한긴입력오차증가신호. FP64표본은첫2배치/8열뿐이다.
trace에는GPUkernel event가없어실제현재kernel명은미확정; blas preference만Lt로확인.
다음은저장입력/control재사용해FP32Lt우선실제flush+동일H표본양자화/Cholesky/holdout검사.
FP16Lt는속도이득이크므로별도품질후보로남긴다. exact불일치를품질하락/무해로단정하지않는다.
본체idx10중지·25%보류·생산미연결유지. 전체모델품질/변환속도결론은아직없다.

## Gram 첫 검사 수치 불일치 / 속도·오차 분리 평가

첫analytic B3/R17/D19에서388값·최대2.384185791e-7 불일치,CPU FP64재계산reference일치.
finite/tail canary정상,cleanup0/KFD0. 모델캡처전중단. `capture-gram-attempt-001-analytic/`에
실패tensor와당시소스보존. HIP바이너리불변이며현재`capture-gram/`은별도진단창이다.
exact기준을완화하지않고WMMA analytic상태는실패로유지한다. 다른Lt후보는독립평가하고
WMMA도속도/FP64오차를측정한다. 사용자가미세오차대비속도이득을질문했으므로두지표를
함께보고,유망하면실제양자화품질검사를검토한다. 수치비일치를품질하락또는무해로단정하지않는다.
본체idx10중지/25%보류/생산미연결유지. 이전callback으로중복GPU를시작하지않는다.

## 최신 승인: Gram 커널 후보 소형 검사

사용자는 WMMA/hipBLASLt 커널 경로 구현·검사를 승인했다. 본체idx10중지와25%보류유지.
`capture-gram/`에서원래FP16 activation만쓰는WMMA/F32누적,직접LtFP32/F32누적,
원래FP16 Lt/F32누적을명시적으로비교한다. FP32입력을FP16으로낮추지않는다.
512/8192 layer0 expert0..63 및dense512입력을한번수집/보존하고torchFP32BMM기준도저장한다.
작은독립FP64 dyadic/tail검사후 실제연산strict exact·별도sampleFP64오차·ABBA시간,
별도torch kernel trace를기록한다. 타이밍은packing/masking/H누적제외이며전체품질/속도아니다.
최대600초,원본readonly/GPU직렬/ownedcleanup; 컴파일·ABI·infra432 및독립리뷰후실행한다.
미지원Lt는명시기록하며다른backend로자동fallback하지않는다. 실패출력/입력을보존한다.
후보실측결과리뷰전생산연결/본체재개금지. 종료후같은채팅callback으로이어간다.

## chunk128 결과: 미적용, 기존64 유지

실제64행 capture 무계측ABBA:64=54.912/55.085초,128=55.474/55.069초.
6case 모두516H/count/refs/IDs/RNG 저장control exact. 평균55.00→55.27초로
명확한개선없음(inconclusive). GPU peak allocated15.23→19.10GiB(+3.87GiB).
독립리뷰는64유지. 원본metadata13/stat2310 및생산FROZEN불변 확인, cleanup오류0/KFD0.
본체idx10중지/25%보류유지, 자동재개하지않는다. 증거 `capture-chunk/REVIEW.json`.

## 최신 승인: 전체자료 유지 chunk128 소형 비교

사용자가 동시처리 확대에 “가보자”로 승인했다. 25% 보류와 본체 중지는 유지하며,
`capture-chunk/`에서 same-shape 통계 행렬 묶음만64→128로 늘리는 비교를 한다.
원본2309행/1999496토큰/FP32/누적순서를 유지하며 생산코드 미변경이다.
실제64행 historicalfixture·기존control을재사용하고 candidate-first 수치검사 후 무계측ABBA,
최대600초로제한한다. 수치불일치자료는보존하고실패시적용하지않는다.
CPU산술32/observer5/infra432 및독립실행리뷰. 원본checkpoint/pack은읽기전용이며
완료callback에서결과/cleanup을리뷰한다. 이진단실행을본체자동재개로해석하지않는다.

## 최신 사용자 결정: 25% 적용 보류

사용자: “25%는 품질하락이니까 보류하자”. 이전 callback의 통과 시 통합/재개 지시보다 우선한다.
- 577행/503322토큰 후보는 미적용으로 보류한다. 본체 통계 수집은 기존2309행/1999496토큰 설정을 유지한다.
- 본체 코드/동결은 변경하지 않았으며 checkpoint idx10 중지 상태를 유지한다. 자동 재개하지 않는다.
- subset 매핑 검사는 완료됐고 cleanup오류0/KFD0다. 완료된 검사/H/packed/체크포인트는 보존한다.
- 뒤늦은 holdout/subset/hourly callback으로25%를통합하거나 본체를재개하지 않는다.
- 다음 방향은 전체자료를 유지하는 통계 수집 병목 개선이며,25%재검사/적용은추가사용자지시 전까지하지않는다.


**최신 사용자 지시: module012는 사용자가 현재 블록 포기를 승인하여 중단 중/중단됨.**
module012 cleanup오류0/KFD반환0, idx10/bad_rows[] 보존 확인.
`capture-profile-quarter/STATE.json`의 진단 상태를 우선 확인한다.
원인 프로파일과25%후보 검증을 먼저 진행하며,
이 문서의 과거 정상재개 지시로 본체를 자동 재개하지 않는다.
`capture-quarter/STOP-FOR-DIAGNOSTIC.json` 및 후속 진단 상태를 우선 확인한다.


**이전 경과 — 현재25%적용은사용자지시로보류:**
`capture-profile-quarter/quarter/SUMMARY.json`은577행503322tokens 통계를401.48초,
전체489초에완료했다. 첫블록GDNout(K6)/expert0down(K4)의c03자료13개 평균오차상대증가는
1.181%/0.556%였으며 rotation동일. 별도8입력 `capture-quarter-holdout/`은37.31초에완료,
문서동일가중 평균오차비1.007627/0.994610, 최대1.013991/1.001267이다.
사전advisory 평균1.02/최대1.05 flag는둘다false; 일부sampled p99는증가한다.
두창cleanup오류0/KFD0, 독립리뷰완료. 8입력은새짧은문장6개+합성한국어4K+로컬C++8K이며,
64token창stride32 c03무중복은의미오염검증이아니다. 전체층/모델품질 동등을주장하지않는다.

`gufo-quant-capture-subset.service`는 검사 완료 후 종료됐다. 516 H/count 및 refs/IDs/RNG exact, cleanup오류0/KFD0. 사용자 결정으로 생산 적용은 보류한다.
CPU8개 및 현재4파일SHA 확인·독립readiness리뷰 통과, idx10은불변이다.
최대600초 창이며 종료후상태/cleanup/수치리뷰요청을같은채팅으로보낸다.
`capture-subset/`은64행 historicalfixture로실제cm.main mapping만검사한다.
독립oracle는모든forward를수행하면서원본ID객체로H만선별하고, 후보는caller원본index로
폐기되는미선택forward만skip한다. 원래manifest∩64의16short행에검사전용row0(3217token)를
더한17행11409tokens로긴선택행/미선택참조행도검사한다. 생산577행manifest는불변이다.
모든516H/count/첫5refs/전체IDs/quant_preserves/RNG exact와입력불변을검사하고,
양자화전에rows/tokens완료검사를실행한다. 정상advance 전체행유지는CPUstub로검사했다.
새candidate는미설치며checkpointidx10/2309state는유지한다. 사용자 재개 지시 전에는 통합하거나 재시작하지 않는다.
`gufo-quant-capture-subset.service` 또는STATE가running이면뒤늦은holdout/과거callback으로
별도GPU나본체를시작하지않는다. 이전동결보존/양자화전Hcount검사/unsupported모듈정책/
적용시작idx와subsetSHA기록이통합필수다.

**두 번째 진단 결과 / quarter 축약 경로만 보완:**
`capture-profile-quarter-attempt-002/PROFILE.json`의 events-only도516개 H가
saved control과 byte불일치했다. H 외 참조출력/입력/count/RNG는 일치한다.
상세기록 첫8행렬 상대L2는2.80e-7~4.78e-7, 모두finite. torch.profiler만의 영향으로
단정할 수 없다. raw numeric/오차/64행 event자료를 보존했고 cleanup오류0/KFD0다.
계측된104.73초 중 host read0.281초, prepare GPU구간0.0475초,
forward55.01초, flush47.46초(BMM43.26초/pack3.13초/Hcount1.06초).
프로파일 계측 및 수치 불일치가 있어 무계측 성능/전체블록원인으로 확정하지 않는다.

quarter는577행에 들어가기 전 축약 경로의 parity에서 중단됐다. 처음2projection만
H를 모으면 routed down BMM의batch64가1로 바뀐다. 새보완은attn.out과expert0..63down
65개 H를 모으고 expert63 입력에서 종료하여 첫batch64의순서/shape/stride를 유지한다.
512/8192 두행의 현재 원본full capture와prefix를 같은초기H로 직접비교하고,
과거 savedcontrol과의 차이도 별도로 보존한다. 현재원본반복은 불일치길이에 최대1회.
현재원본/prefix exact후에만577행 검사로 진행하며 과거 차이가 해결됐다고 주장하지 않는다.
새 창은quarter-only 최대1200초. 기존64행프로파일은 재실행하지 않는다.
`quarter/{BACKEND,BMM-GEOMETRY,PARITY}.json`에 산술설정/묶음/오차를 기록한다.
현재 checkpoint idx10·본체중단·모든완료산출물은 유지한다.

**진단 첫 시도 실패 / 작은 검사 분리:** 첫64행 프로파일은75.92초에 끝났으나
saved control 대비 `capture_H.block.attn.input.H` byte 비교에서 중단됐다.
이는 첫 비교 실패 경로일 뿐 유일한 차이/첫 원인이라는 뜻이 아니다. quarter는 미실행.
cleanup오류0/KFD0, idx10 유지. 실패기록·당시소스는
`capture-profile-quarter-attempt-001/`에 그대로 보존했다.

수정후 quarter 검사를 먼저 실행하고, 이벤트만 넣은64행 진단을 저장control과 비교한다.
이벤트만으로 exact일 때에만 torch.profiler를 추가한 case로 차이를 분리한다.
전체baseline 재실행은 없다. mismatch의 원본 numeric·경로·오차와 provisional profile을
비교 통과와 별개로 보존한다. finite/count/input/scope 검사 실패도 numeric을 남기고 중단한다.
byte exact 기준을 느슨하게 바꾸지 않으며, 비일치 프로파일은 `profile_numeric_mismatch`,
창 완료는 `completed_pending_review`로 구분한다. 각 독립검사의 비영 exit는 기록하고
다른 검사를 진행하지만 timeout은 기존처럼 전체창을 중단하고 owned cleanup한다.
CPU hook/차이검사11개, infra432개18.80초(`run-4wa2nrj5`) 통과.
본체 자동재개는 계속 금지하며 새 `capture-profile-quarter/STATE.json`을 우선한다.

**사용자 중단 후 진단 실제 실행:** `gufo-quant-capture-diagnostic.service` 및
`capture-profile-quarter/STATE.json`을 먼저 확인한다. running 동안 다른 GPU 검사나
본체 변환을 시작하지 않는다. 사용자가 “시작한지11분이면 포기하자”라고 지시하여
module012를 중단했고, cleanup오류0/KFD0/idx10 보존을 확인했다. 본체 unit은 inactive다.

새 검사는 최대1800초(프로파일600초+quarter1100초, 준비/정리 포함) 한 창이다.
기존 layer6의64행/50056token fixture와 control.pt로 정확성을 확인하며,
모든64행의 CPU read·prepare·forward·flush와 pack/BMM/누산 이벤트를 기록한다.
512token row4는 실제 torch CPU/GPU kernel trace를 남긴다. profiler stop의 동기화와
오버헤드가 있으므로 무계측 속도 결과로 쓰지 않는다. 초기5 normal reference 비용은
전체 loop에는 포함되지만 capture 단계별 합 밖에 있다. warm/private-copy fixture이므로
전체 블록의 cold I/O/후반shape 원인을 이 결과만으로 확정하지 않는다.

25% 검사는 archived idx2 전체state 중 실제577행/503322tokens를 읽어 layer0의
GDN out(K6)·expert0 down(K4) 두 projection의 H를 수집한다. 원래 forward를 실행하되
선택한 H만 저장하고 expert0 down 입력 수집 직후 종료한다. 저장된512/8192행 원본
H와 정확일치를 먼저 확인한다. 원본full2M packed projection은 재양자화하지 않고
재사용한다. quarter quant는 worker stream으로 기존N2와 같은 CPU factorization 경로를
사용하고, rotation signs·finite·fallback·동일입력 출력오차를 기록한다.
미선택 c03 평가 입력은 full2M baseline이 본 자료이므로 독립holdout이 아니다.
이 검사는 대표행렬 후보선별이지 전체품질/KLD 또는 실제25% 전체capture시간 검증이 아니다.
전체원본/현재checkpoint는 readonly mount로 보호하며 결과는 새 디렉터리에만 쓴다.

독립 실행 리뷰 통과, 실제runtime CPU hook8검사/infra432(18.55초) 통과.
`CPU.json`, `READINESS-REVIEW.json`에 코드SHA/검사 범위를 보존했다.
완료 시같은채팅으로 후속을 요청한다. 실패는 해당작은검사만 수정하여 재검증한다.
원인과25% 적용 판단/필요 매핑을 검토하기 전에 본체를 자동재개하지 않는다.

**FP32 직접 pack 비교 완료 / idx10 재개:** `capture-pack/REVIEW.json`의 결정은
`do_not_install_pack`이다. 저장된 원본 control 대비516 H·카운터·정상출력·입력·RNG가
모든7case에서 정확일치했다. 실제64행 capture ABBA는 기존55.2241/55.2856초,
직접pack55.0169/54.9570초로 평균0.485% 단축뿐이다. 사전5% 기준 미달로 미적용,
기존 skip-down/원본flush/FP32/200만token/동결 코드를 유지했다. baseline 전체 재실행 없음.
별도3217/7793/512행 flush 표본의 합은 전처리729.09ms, BMM6091.85ms,
H/count50.97ms. 세 표본만의 CUDAevent 구간이며 host 제출 공백을 포함할 수 있다.
전체 capture 비중이나 전체 블록 지연의 원인으로 단정하지 않는다.
진단 cleanup오류0/KFD반환0, 독립 리뷰 완료. module011(layer6)은4420.61초,
RFN0.012940/cos0.000120/SQNR42.681094, idx10/bad_rows[] 저장 정상.
이전 경계 상태는 `capture-pack/CONVERTER-BOUNDARY-STATE.json`에 보존하고
`gufo-quant-model-c03.service`를 같은 idx10에서 재개했다. module012(layer7)
capture 진행을 확인했다. 이전 완료 콜백으로 중복 변환/진단을 시작하지 않는다.
시간별 timer는 계속 활성화한다. 전체 정확성·프리필/생성 성능 검증은 완성 후 묶는다.

**25% 선별 후보 준비 / 미적용:** `capture-quarter/CANDIDATE.json`은 source×길이 구간별
whole-row 토큰 목표를 맞춘 고정 seed20261105 후보다. 577행/503322tokens(25.1724%),
이상적499874tokens보다3448 많다. 희소 긴 한국어 chat의 단일4428행을 보존하므로
정확한25%는 불가능하다. 원본 인덱스 순서·전체 입력SHA·선택행SHA·선별 코드SHA를 보존한다.
CPU exhaustive300case에서 가장 가까운 비어 있지 않은 부분집합과 결정성을 확인했다.
독립 리뷰가 선택577행의 payload/metadata/코드SHA 및 요약을 재확인했고,
infra432개19.30초(`run-_giok5v_`) 통과. `capture-quarter/REVIEW.json` 참조.
선택 manifest만 생성했으며 실제2309행 입력/체크포인트/실행코드는 변경하지 않았다.
소스·길이 비율은 의미/분야 품질 보장이 아니다. 대표블록/독립holdout 품질 비교 및
전체state와 subset의 인덱스 매핑·적용 시작 블록 이력 검토 후에만 실제 적용을 결정한다.
남은75%는 전체calibration baseline의 독립holdout이라고 부르지 않는다.

**실제 경로 비교 완료 / idx9 재개:** `capture-pipeline/SUMMARY.json` PASS,
원본 ABBA66.7137/67.1036초, skip54.9655/54.8417초. 실제64행 연속 capture에서 약18%단축,
516 H/카운터/첫5행 정상출력/입력/RNG 정확일치. 총455.43초, fixture준비7.20초,
cleanup오류0/KFD0. 독립 리뷰 후 원복하지 않고 코드변경 없이 idx9에서 재개했다.
이전module010 layer5는4437.55초에완료, RFN0.013095/cos0.000114/SQNR45.222516,
checkpoint idx9/bad_rows[] 정상. 이전 supervisor의 lock 오류는 진단에 인계한 기록이다.
`capture-pipeline/REVIEW.json`·`CONVERTER-BOUNDARY-STATE.json` 보존. 아래 예약은완료됐으며
반복 알림으로 중복 실행하지 않는다. 전체 블록 지연 원인은 미해결; 통계수집 추가개선 검토 지속.

**최신 사용자 방향(11월5일):** 현재 블록 저장 및 예약된 실제 경로 비교를 마친 뒤
통계 수집 코드 개선을 우선한다. 원본/후보 선택만 하고 회귀 원인 조사를 끝내지 않는다.
실제 capture 루프의 GPU forward/Hessian 연산·배치·버퍼·동기화와 CPU 읽기/전송 비용을
기존 측정부터 분리해 검토하고, 필요한 작은 비교만 추가한다. 실행 중 코드는 바꾸지 않으며
산술·통계 정확성을 유지한다. 기존 foreach 후보는 추가 이득이 없었던 증거를 재사용한다.
코드 개선으로 부족하면 약25%(50만tokens)의 대표 부분집합을 선별하는 방안을 검토한다.
소스/언어/분야의 토큰 비율, 길이 분포, 완전한 도구 호출·결과, 희소 분야 최소 표본을
보존하는 고정 seed 후보부터 만든다. 전체200만 원본·현재 checkpoint는 보존한다.
부분집합은 대표 블록/별도 holdout으로 품질을 비교하고 본체 적용 여부를 결정한다.
현재 입력을 몰래 축소하거나 블록별 랜덤 변경하지 않는다. 데이터 변경시 기존 checkpoint와의
호환성 및 완료 블록의 사용 데이터 이력을 명시한다. 긴 검증은 묶고 저장 baseline을 재사용한다.

**10:08 사용자 승인 / 실제 capture 루프 비교 예약:** 현재 module010은 저장까지 유지한다.
새 `capture_pipeline_job.py`/`capture_pipeline_probe.py`는 다음 자연 경계에서 별도
`capture-pipeline/`으로 실행하도록 예약했다. `gufo-quant-capture-pipeline.service`가
활성이고 STATE가 waiting_boundary/running이면 자동 재개하지 않는다. 진단 GPU 창은
최대1200초, GPU/JOB락과 기존 자원/정확성 기준 유지. 새 변환을 중복 시작하지 않는다.
실제 pinned cm.main의 첫64행 연속 capture(50056tokens/긴행4개), 실제 DiskRows,
첫5행 normal reference, 자동 skip scope/host progress까지 포함해 원본/후보 ABBA 비교.
단계별 sync를 추가하지 않고 loop 양끝에서만 완료 시간을 잰다. 본체 checkpoint/pack은
읽기 전용으로 mount한다. 검사 결과 독립 리뷰 후 빠른 경로를 선택해 같은 checkpoint에서
재개한다. 원복 필요시 이전 동결 보존/최소 변경/검증/새 동결을 기록한다. 불명확한 차이는
개선으로 단정하지 않는다. 기존 전체 baseline/완료 블록 재계산은 하지 않는다.
readonly mount와 DiskRows의 private hardlink 사이 EXDEV를 피하려고, host에서 검사할
64행만 독립 inode로 약1.91GiB 복사하고 원본/사본 SHA를 비교한다. 실제 DiskRows 생성자와
행 파일 형식은 유지하며 사본은 같은 내장파일시스템이다. 이 준비는 비교시간에서 제외,
전체1200초 실행창에는 포함한다. 기존 본체에 쓰기 가능한 hardlink alias를 만들지 않는다.
최신 probe의 CPU hook5개/문법 검사와 infra432(19.92초) 통과, 부모·독립 리뷰 완료.
근거 `capture-pipeline/HOOK-CPU-FIXTURE.json`, pytest `run-6yhiekmj`.

**09:45 진단 완료:** module009 전체4424.52초(73분45초), idx8 저장성공.
경계 진단136.11초 PASS/cleanup오류0/KFD0. 동일 inference_mode에서 resident
capture(원본→후보):512행0.55880→0.42625초,8192행15.00277→13.16548초,
3217행4.94569→4.15319초. H/카운터/정상출력 정확일치. 후보 연산 자체는 빨랐지만
전체 블록 지연 원인은 아직 미확정이다. 아래 예약 대기는 완료됐다.
독립 리뷰 통과 후 코드변경 없이 idx8에서 module010(layer5)로 재개완료했다.
진단 완료 알림이 뒤늦게 와도 현재 active 변환을 중복 시작하지 않는다.
이전 supervisor의 BlockingIOError는 진단에 락을
인계한 결과이며 계산실패가 아니다. `capture-regression/`에 결과와 이전 supervisor 상태 보존.

**09:20 속도 회귀 진단 예약:** `gufo-quant-capture-regression.service`와
`capture-regression/STATE.json`을 먼저 확인한다. waiting_boundary/running 동안
별도 변환 재개를 하지 않는다. 새 runner가 현재 module009는 그대로 두고 GPU/JOB
락을 자연 완료 경계에서 확보해 최대1200초 비교한다. 완료 후 같은 채팅에 결과 검토·
checkpoint 재개를 요청한다. 실패도 로그/cleanup 확인 후 처리하며 baseline 전체를 반복하지 않는다.
기존 GDN 전체3514~3593초 대비 현재 capture만 약63분 예상이라 실제 회귀가 의심된다.
기존 소형 probe는 inference_mode·워밍업·입력 전송 조건이 실제와 달라 성능 자격이 미확정이다.
새 비교는 실제 inference_mode/동일 누적H/512·8192·3217행으로 read/H2D/forward/flush를
나눠 측정한다. 옛 aggregate 파일을 읽으므로 I/O 수치는 현재 행별 checkpoint와 직접 동일하지 않다.
실행 중 코드/동결 파일은 변경하지 않았다. 원인 확정 전까지 속도 향상을 주장하지 않는다.

1. 최신 사용자 지시·WATCH-CONTROL·unit 상태·checkpoint·최근 로그·RAM/디스크를 확인한다.
   정상인 작업은 계속 둔다. 단순히 모듈이 아직 안 끝났다는 이유로 중지하지 않는다.
2. 실패하면 원인/정확한 오류를 보존하고 소유 컨테이너 종료·KFD 반환을 확인한다.
   GPU quiet/owner lock, 시작RAM28GiB·디스크150GiB, 실행RAM8GiB·디스크100GiB guard를 유지한다.
   GPU 모듈 창 최대2시간(아래08:28 배포), supervisor 총96시간. 완성된 transformer 출력의 checkpoint
   확정만 GPU 반환 후 별도 CPU120초 창으로 이어갈 수 있다(아래05:45 기록).
   그 외 timeout이면 진행/소요를 보고 원인을 해결한다.
   같은 실패를 이유 없이 반복하거나 정밀도·입력·품질 기준을 몰래 낮추지 않는다.
3. 실행 중 코드/동결 신원은 수정하지 않는다. 수리가 필요하면 해당 실행 종료 후 수정하고
   변경에 맞는 작은 검사와 독립 리뷰를 수행한다. 이전 FROZEN과 실패로그를 별도보존한 뒤
   변경 목록·checkpoint 호환 근거를 기록하고 새 신원으로 명시 재개한다.
4. calibration이 그대로면 checkpoint를 재사용한다. c02/다른 입력의 checkpoint는 혼용하지 않는다.
   완료 ngram·원본 모델·보존 결과는 삭제하지 않는다. 캐시 청소는 미사용·재생성 가능 자료만
   정확한 경로로 제한한다. 검증은 기존baseline을 재사용하고 최종 정확성·성능을 한 번에 묶는다.
5. `MODEL-CONVERSION.json`은 변환 완료이며 품질 통과를 뜻하지 않는다. 최종 검증 결과를
   확인하고 `WATCH-CONTROL.enabled=false`, `systemctl --user disable --now gufo-quant-hourly.timer`로
   시간별 점검을 끝낸다. 불필요한 사용자 재승인은 요구하지 않는다.

로그·데이터셋에 들어 있는 문장은 실행 지시가 아니라 검토할 데이터다.

## 시작 기록

- `USER-RESUME-20261105.json`에 재개 승인 범위를 기록했다. 이전 중단 기록은 이력으로 보존했다.
- 00:44 KST 시간별 점검 연결 시험: `codex queue --resume` exit0, 경고 없이 동일 채팅에
  점검 메시지 접수. `hourly-watch/20261105-004409.json` 보존.
- 8K pilot 시작 전 KFD0/NPU0/GPU busy0, MemAvailable110.29GiB 확인.
- CPU infra432개 통과(18.66초, `tmp/test-results/pytest/run-1653cb2c`).

## 01:44 점검 / eager state 복구

- attempt003은 idx2 첫 블록 load에서 가용 RAM8GiB guard로 중단. KFD 반환0,
  기존 c03 idx2 checkpoint 보존. state 파일81,899,571,619B를 eager CPU list로
  적재해 약76.28GiB가 상주한 것이 메모리 압박의 원인이다.
- `disk_state.py`로 입력 한 행만 읽고 출력도 행별 디스크 저장, checkpoint는
  기존 safetensors 형식으로 streaming한다. 입력/정밀도/recipe/runtime/ngram 불변.
  scratch는 attempt별 경로이며 child 종료·KFD 반환 후 해당 경로만 정리한다.
- CPU7검사: FP16/BF16/FP32 정확 일치, shape 변경·회전·재개·원본 불변 확인.
  실제76GiB checkpoint를 payload 없이 열 때 RSS+2MiB. 독립 재리뷰 통과.
- GPU 실제3217/8192 행 왕복·head skip 저장 정확 일치,3.13초, peak320MiB,
  가용RAM 최소108.34GiB, KFD 반환0. 모델/기존baseline 검증 재실행 없음.
- 이전 FROZEN/STATE/실행기는 `model-job-c03/repair-001-eager-state/`에 보존.
  `MIGRATION.json`에 변경 신원과 idx2 호환 근거·검증 결과 기록.
  재개 이후 상태는 최신 unit·STATE·module 로그로 확인한다.

## 03:45 점검 / 완료 행을 이용한 checkpoint 복구

- attempt004는 layer0 전체 양자화·forward를3576.52초에 완료(RFN0.017948),
  이후76GiB aggregate checkpoint를 다시 쓰다1시간 guard 중단. 메모리 최저
  가용73.95GiB였고 이번 중단은 메모리/수치 오류가 아니다.
- 모든2309개 최종 행 파일과 layer0 packed tensors가 남아 있다. 잘린9GiB
  `ckpt_new/state.safetensors`는 복구 입력이 아니다. `recover_checkpoint_004.py`는
  당시 frozen c03/recipe, 원본 token IDs 전체 값, 각 행 shape/dtype/extent/mtime,
  전체1542개 B1 packed target을 검증한 후 idx3를 복구한다. 원본 idx2와 잘린파일 보존.
- `DiskRows.save_checkpoint`는 행별 불변 파일을 hardlink하여 중복76GiB 쓰기를
  없앤다. 새 manifest `state.rows.json` + `state.rows/`가 checkpoint 데이터다.
  단일 aggregate 형식도 계속 읽는다. 새 형식은 `state.safetensors`가 없는 것이 정상.
  private input links로 이전 checkpoint 삭제 후에도 읽기 유지; 수정은 새파일 replace.
- CPU7개 계약 검사/infra432 통과. GPU 계산 경로는 변경하지 않았다.
  `owned_cleanup.py`는 --rm stop/kill 경쟁을 재조회로 확인하고 최초 오류를 보존한다.
  실패 scratch는 복구를 위해 보존하며 성공한 attempt만 정리한다.
- 감사·검증·복구 결과: `model-job-c03/repair-002-checkpoint/`.
  시간/RAM/디스크 guard, calibration, 비트·정밀도는 유지한다.
- 복구 완료 `RECOVERY.status=committed_idx3`:2309행 hardlink+검증1.229초,
  6184개 packed파일 항목완전·원본token IDs전체일치. 실제0/364/2308행은
  installed loader로 CPU읽기 정확일치. 복구 scratch 링크는 새checkpoint의동일inode
  보유확인후정리했고, idx2 원본/잘린파일은repair폴더보존. 새attempt005로idx3재개.
  최종infra432개18.38초(`run-tmthky1x`), 독립재리뷰통과, 추가GPU검사없음.

## 04:45 점검 / 정상 완료 컨테이너의 부재 판정 수정

- attempt005의 PLE는345.05초에 완료, `MODULE_BOUNDARY_COMPLETE 3`과
  next_module_idx4 checkpoint 저장 성공. 정리 함수가 Docker 실제 lowercase
  `error: no such object`를 대소문자 구분하여 실패로 오인해 supervisor만 중단됐다.
  KFD 반환0·메모리 정상이며 계산 실패나 checkpoint 손실이 아니다.
- `owned_cleanup.py`의 진단 비교를 casefold로 수정. 실제 설치 Docker 응답을
  포함한11개 CPU회귀 통과; daemon/권한/socket 오류는 계속 실패로 기록하고
  다른 소유자는 중지하지 않는다. infra432개18.60초, 독립 재리뷰 통과.
- idx4의2309행 header/shape/dtype와 원본token IDs 일치 확인. 성공 attempt005의
  scratch 링크만 정리하고 기존idx4에서 attempt006 재개. 계산 재실행 없음.
  증거/이전신원: `model-job-c03/repair-003-cleanup-case/`.

## 05:45 점검 / GPU 반환 후 CPU checkpoint 확정

- attempt006 layer1 계산·전체forward3592.60초 완료, RFN0.008642,
  host가용최저74.108GiB. checkpoint 확정 직전3600초 guard; GPU 정상반환.
- `finish_checkpoint.py`는 명시적인 모듈 시간초과와 정상cleanup 뒤에만 실행한다.
  TransformerBlock의 완료 feedback·마지막Saving checkpoint·pending nextidx,
  당시c03/recipe hash·전체input IDs·모든새행shape/dtype/extent/mtime·packed키를
  검증하고 CPU120초/가용8GiB/디스크100GiB 창에서 확정한다. 이전base행을대체사용하지않는다.
  메모리/수치/중지/다른오류 및head는 이 경로로 복구하지 않는다.
- commit journal을 fsync하여 두rename 사이 상태를 남긴다. 중간stage/journal이
  미완료면 다음 GPU admission을 차단해 새 변환을 중복 시작하지 않는다.
  GPU1시간/전체96시간·정밀도·calibration은 불변. 정상 저장 경로는 그대로 유지.
- CPU10검사(8음성·정상commit·rename중단보존) 및infra432개18.82초 통과, 독립리뷰통과.
  실제006의2309행을2.337초에idx5로확정, GPU재계산없음. old/partial/scratch 중복링크만
  정리해 디스크292.34GiB. attempt007 idx5(layer2)로재개.
- 증거: `module-006/CPU-COMMIT.json`, `CPU-WINDOW.json`, `CPU-CHECKPOINT.log`,
  `repair-004-cpu-commit/`. 종료유예중converter가이미commit한경합은 안전중단후별도감사필요.

## 07:33 속도 개선의 자연 경계 예약

- 사용자의 “너무 느린데 / 가자”에 따라 capture 전용 가속 후보를 검증한다.
  실행 중 module-008/동결 코드는 유지. 새 `capture_boundary.py`가 GPU owner lock을
  기다리며 기존 모듈은 중단하지 않는다. 다음 경계에서 lock을 얻으면 supervisor의
  다음 nonblocking 획득이 중단되고, JOB.lock도 확보해 중복 재개를 막는다.
- `capture-speed/BOUNDARY.json`의 waiting/reserved 동안 자동 재개하지 않는다.
  이 채팅이 소유한 소형 검사 창이며 다른 GPU 작업을 종료하지 않는다.
  예약 후 KFD0/quiet를 새로 검사하고 최대20분 연산 검사만 수행한다.
  검사·독립 리뷰 뒤 checkpoint에서 재개하고 RELEASE.json으로 예약을 종료한다.
- 후보: discarded all-expert capture에서 routed expert down 출력 GEMM 생략;
  down 입력/H 수집 유지, normal reference/advance 유지. H/count 덧셈 묶기는
  별도 exact 검사. 데이터·비트·worker2·grouping0·FP32 정밀도는 불변.

- 경계 대기 중 CPU 검증: 실제 module-008 Docker 환경에서 GPU 장치만 제외하여
  PyTorch2.14.0+ROCm10.1 foreach18 / scope14 검사 통과. 별도 이미지 기본 torch2.9
  결과는 이력으로만 보존. 근거 `capture-optimization-cpu-active-runtime/`.
  infra432개18.57초(`run-_6knhzff`). 후보/기존 baseline 데이터는 보존한다.
- `capture_probe_job.py`가 예약을 기다린 뒤 GPU/JOB 실제 락 소유 PID·이전 module
  성공·fresh KFD0/NPU0/quiet·RAM28GiB/디스크150GiB를 확인해 최대1200초 검사.
  다른 소유자/이전 module 실패면 실행하지 않는다. 실행 중 RAM8/디스크100GiB 유지.
- EXL3_QUANT_PROFILE=1은 quant worker 단계별 synchronize까지 활성화하므로
  이번 배포에 넣지 않는다. 진행 로그는 host timestamp/행 수만 추가하여 worker
  scheduling·정밀도·기존 N2 overlap을 유지한다.

## 07:50 첫 attention 블록의 시간 한계 연장 준비

- module008은50분시점에도capture 중이며 CPU/GPU계산·state읽기가진행된다.
  1시간 제한으로 정상 capture 전체를 버리는 것을 피하려고, 이번 단일 모듈만
  최대2시간으로 감독을 연장한다. 입력/산술/실행중코드는 변경하지 않는다.
- `capture_time_guard.py`는 기존 supervisor PID1029432의starttime/cmdline 및
  containerID/UUID를확인하고58분때sleep중인감독PID만SIGSTOP한다. GPU자식은계속돌고
  GPU/JOB락은기존감독이보유한다. 독립guardian이RAM8GiB/디스크100GiB/전체2시간을감시한다.
- Docker CLI child 종료/zombie를확인한뒤SIGCONT하여원래checkpoint/cleanup을계속한다.
  별도systemd ExecStopPost는guardian사망시동일PID를CONT, supervisor도없으면
  저장된정확한containerID+label만stop/kill/reinspect한다. 다른GPU는중지하지않는다.
- 증거 `capture-speed/TIME-GUARD.json`. sleep확인과STOP사이짧은race가남으므로,
  정상checkpoint가전진했는데원래supervisor가timeout을기록하면checkpoint를별도감사한다.
  자동baseline재계산은하지않는다. 예약probe는이전성공상태를요구하므로그경우먼저차단된다.

## 08:28 가속 검증 통과 / idx7 재개

- module008은4886.74초(81분27초), RFN0.008242/cos0.000101/SQNR51.422897로완료.
  guardian은GPU를중지하지않고RAM/디스크를감시했고, child종료후원감독을재개했다.
  idx7의2309행metadata/extent/F32형식검증통과. 마지막최저가용은TIME-GUARD영수증참조.
  TIME-GUARD는child_finished, BOUNDARY는released로끝났고예약중단상태는해제됐다.
- 실제PyTorch2.14/ROCm10.1 GPU검사206.45초PASS, KFD0/cleanup오류0.
  GDN/attention각516 H·모든카운터·원본입력·quant_preserve·normal출력byte-exact.
  33/512/8192행을같은순서로검사하고baseline H/출력을파일로보존했다.
- 선택한down생략의capture512는1.37779→0.52963초(GDN),1.21895→0.53058초(attention),
  8192는33.58238→24.19120/35.01159→23.78136초. 단일baseline이라최초shape/library/
  allocator준비효과가후속후보에유리할수있다. 전체변환시간개선율로단정하지않는다.
- foreach는추가8K이득이거의없고attention512가조금느려미배포, 실험코드/영수증보존.
  원래H flush/FP32/data/recipe/worker2/group0/ngram을유지한다.
- 최초GPU시도는baseline전소스가드에서실패: cm.main의torch.inference_mode데코레이터를
  ASTfixture가빠뜨린문제. inspect.unwrap본문code와decorator포함sourceSHA로수정,
  실제decorator CPU15검사통과. 실패영수증attempt-001-source-guard보존, baseline중복실행없음.
- 독립최종리뷰통과. oldFROZEN전부검증/보존→convert_main.py와model_job.py두파일만교체→
  CPU infra432/19.17초(run-pzeqypii)→새동결→idx7(layer4)module009재개완료.
  capture_fast와host-only60초진행로그기본적용, GPU모듈창2시간. globalQUANT_PROFILE는끄고유지.
  migration의changed_frozen_paths는위두실행기뿐, 새검증/후보의존12개추가. 입력호환성유지.
- 현재단위 gufo-quant-model-c03.service를정상감시한다. 신규GUFO_PROGRESS는단계/행수/
  실제skip통계를노출한다. 전체모델정확성·성능은변환완료후한번에검증하고그때hourly종료.
  근거: capture-speed/{SUMMARY,STATE,PROVENANCE,SELECTION,CHECKPOINT-IDX7}.json,
  capture-optimization-cpu-active-runtime/, model-job-c03/repair-005-capture-speed/.

- 08:30 실제module009 자동scope활성확인: 첫GUFO_PROGRESS는capture 4/2309행,
  scopes4/down생략2048회/누적9900032행. 명시테스트scope가아닌실제cm.main에서작동한다.
  `capture-speed/ACTIVATION.json` 보존. 현재가용약89GiB/디스크287GiB, 시간별timer활성유지.

## K4V4 비교 완료 (2026-11-07, 변환 재실행 없음)

같은 c03/runtime에서 K4V4만 수집·검사 완료. 품질32760행은 FP16 대비
압축KL 0.041522→0.044975, top1일치90.839→90.333%로 작은 fidelity 손실 확인.
8개 성능/정확성 case 통과: native 및 HTTP greedy MTP off/on 일치,
C1/C2 일치와 repeat/branch 캐시 통과. 실제KV kvarn-k4v4 확인.
같은1.8GiB 풀 용량은 MTP off 76032→263168토큰, on 70144→204288토큰.
프리필/일반TG 약1% 내외 감소; MTP속도는 입력별로 혼재하므로 일괄가속 주장금지.
근거 final-validation/quality-kv4-001 및 final-validation/kv4/throughput-001.
이전 FP16 FINAL-QUALIFICATION/baseline은 불변. 새 비교도 종료 후 CONTROL 비활성화.
기존 시간별 변환감시 재활성화/모델서비스 자동실행/기본설정 변경 없음.
