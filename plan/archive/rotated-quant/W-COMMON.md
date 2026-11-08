# RQ 작업 흐름 공통 규칙 (W1–W7 에이전트용, 2026-10-29)

모든 W 브리프는 이 문서와 [R-BRIEF.md](R-BRIEF.md) "공통 규칙"을 따른다. 두 문서가 다르면 이 문서를 따른다. 전체 그림은 [HANDOFF.md](HANDOFF.md), 설계는 [PLAN.md](PLAN.md)에 있다.

## 감독과의 연락

- 감독은 메인 세션(gufo-06)이다. 묻거나 허락을 받아야 하면 `SendMessage(to: "main")`로 보낸다. 감독의 답을 기다려야 하는 지점에 오면, 그때까지의 상태를 보고하고 거기서 멈춘다. 보고 첫 줄에는 `WAITING: <무엇을 기다리는지>`를 쓴다. 감독이 SendMessage로 작업을 이어 준다.
- 다른 세션(gufo-00, gufo-d7)에는 직접 메시지를 보내지 않는다. 장비 조율은 감독이 한다.
- 결정이 필요한 일(계획의 전제를 바꾸는 결과, 범위 밖 변경, 공유 파일 수정)이 생기면 멈추고 감독에게 보고한다.

## 장비

- **무거운 CPU 작업**(빌드, 모델 변환, 큰 파일 해시, 큰 numpy 계산, 10 GB를 넘는 디스크 쓰기)은 **감독의 허락을 받은 뒤**에만 시작한다. 감독은 CPU 단계 동안 `ABBA-OWNER.lock`을 직접 잡고 있다. 그러니 에이전트는 이 lock을 잡지 않는다(잡으면 다른 W와 충돌한다). 허락을 받으면 그 단계 안에서 여러 명령을 이어서 돌려도 된다.
- **GPU 창**은 감독의 허락을 받은 뒤에만 시작한다. 창은 `window.py`가 lock을 잡으므로, 감독이 자기 lock을 놓은 뒤에 허락한다. 창 하나에 한 번만 돌리고, 실패해도 자동 재시도하지 않는다. 그 밖의 절차는 R-BRIEF와 K1 브리프의 "공통 규칙" GPU 항목을 그대로 따른다. 창 계획은 `~/projects/ai.tools/test/experiments/gufo_upstream_port_20261023/PLAN-RQ*.json`에 쓴다.
- 가벼운 일(소스 읽기, 작은 테스트, 작은 Python 스크립트)은 허락 없이 해도 된다. 다만 GPU 창이 도는 중(`docker ps`에 창 컨테이너가 보일 때)에는 멈춘다.
- **컨테이너 이미지:**
  - CPU 작업, Python, 단위 테스트, ROCm 7.2.4 gpu-test 빌드: `ai-tools/gufo-eval:7c2c5a728270`
  - 운영과 같은 ROCm 10.1 / Clang 24 측정 빌드: `ai-tools/gufo-eval:05e527d2cae5-rocm101`. F23 창(`PLAN-F23.json`)이 이 이미지를 썼다. 측정 빌드를 만드는 방법은 그 창의 기록에서 확인해 맞춘다.
- 모델 디렉터리를 컨테이너에서 쓰려면 `-v /home/m2on/projects/ai.tools/storage/models/llm:/models` 같은 마운트를 더한다(원본은 `:ro`로 마운트한다).
- 빌드 디렉터리는 W마다 따로 둔다: W1은 `build/rq-llama/`, W2는 `build/rq-w2-*`, W3은 `build/rq-cpu-test`·`build/rq-gpu-test`·`build/rq-release`. 기존 `build/cpu-test`, `build/gpu-test`는 다른 세션이 쓰므로 건드리지 않는다.
- 루트 여유 공간이 150 GiB 밑으로 내려가면 창이 시작되지 않는다. 큰 파일을 만들기 전에는 `df -h /`를 확인하고, 실험 파일은 측정이 끝나면 지운다. 다만 지우기 전에 감독에게 확인을 받는다.

## 커밋

- 자기 W의 파일 경계 안에 있는 파일만 커밋한다. 다른 에이전트와 같은 인덱스를 쓰므로 `git add` 뒤에 따로 `git commit`하지 않는다. 반드시 한 명령 줄에서 경로를 지정해 커밋한다: `git add <경로...> && git commit -m "<type>(<scope>): <한 줄>" -m "<트레일러>" -- <경로...>`(새 파일은 `git add`가 먼저 필요하다)
- 한 줄 Conventional Commit에 아래 트레일러 두 줄을 붙인다. push, PR, 이슈는 하지 않는다.
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01HuCQEe5fZ9Avb8GWLR598A
  ```
- C++/HIP 파일은 커밋하기 전에 포맷 검사를 통과시킨다(R-BRIEF의 포맷 명령. nixos/nix 컨테이너를 쓰므로 무거운 CPU 작업으로 친다).
- 다른 세션의 파일과 untracked 파일(`.dockerignore`, `.gufo-revision`, `plan/archive/gufo-upstream-port/*`)은 건드리지 않는다. `PLAN.md`, `HANDOFF.md`, `RESULTS.md`의 상태 열은 감독이 고친다. 에이전트는 자기 `*-RESULTS.md`만 쓴다.

## 보고

끝날 때(또는 WAITING으로 멈출 때) 다음을 보고한다.

- 바꾼 파일과 커밋
- 돌린 검사와 결과(실패를 포함해 원문을 요약)
- 측정 표와 원시 결과의 위치(RESULT 행, WINDOW.json, 해시)
- 남은 문제

짐작은 짐작이라고 적는다.
