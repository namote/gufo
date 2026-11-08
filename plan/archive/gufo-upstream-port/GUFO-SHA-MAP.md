# gufo 커밋 해시 대응표 (2026-10-24)

사용자 지시로 gufo 로컬 브랜치(`ai-tools/port-*`, `cache-s4`, `paged-kv*`, `shared-prefill`, `upstream-0.21-merge`)의 작성자·커미터 정보를 바꿨다. 코드(tree)는 그대로이고 커밋 해시만 바뀌었다.

문서·`PLAN-*.json`·LOCKS·이미지 태그(`ai-tools/gufo-eval:<옛 해시 12자>`)·이미지 안 `.gufo-revision`은 옛 해시를 쓴다. 옛 커밋은 저장소에서 지웠으므로 이 표로 찾는다.

|옛 해시|새 해시|제목|
|---|---|---|
|`d42b292c52b4`|`05ffd3406c37`|perf(qwen27b): share one pinned prompt-feature capture across slots|
|`852b6c722985`|`060b8b3e4e39`|perf(serve): checkpoint where the leading system messages end|
|`9a5cacee5e48`|`3ed6a27edb8c`|perf(serve): keep tail checkpoints before the end of long prompts|
|`17ef432631a3`|`eff2179abc92`|perf(serve): skip a stop at a stable boundary near the reused frontier|
|`3c600e8ac68b`|`f2edd14257b5`|test(qwen27b): lease one transfer stream per load in the retained-block restore case|
|`833246db4abd`|`6f86d9ff198f`|perf(cache): checkpoints retain shared-pool KV blocks instead of copying rows|
|`3942939b8a87`|`a08a982e560a`|fix(cache): a checkpoint covers only the frontier it is a prefix of|
|`85890ae19bf1`|`e91544637b69`|fix(cache): know a refused capture before prefill without in-pass prediction|
|`72e451b7cb86`|`80e5ab90edd9`|fix(cache): keep live frontiers on disk across shutdown and reassignment|
|`19c7e0193a31`|`17df9dec4469`|perf(cache): reuse chunks of RAM-persisted checkpoints in later disk captures|
|`82327bd4dfa3`|`095fc6495beb`|fix(cache): write back sessions idle while others run; keep prefill stops while a restore settles|
|`76a731de13d1`|`ba954637f2c4`|perf(cache): write idle live frontiers to disk and skip prefill stops the cache would refuse|
|`3e36130a4c9a`|`0ec672e58bd9`|perf(cache): warm restore files with real reads instead of WILLNEED hints|
|`2bae00dc5b64`|`bec25f2cbc2a`|fix(serve): log a reassignment capture refused for RAM capacity as refused, not failed|
|`18b9e8865145`|`1bb53f8bf743`|perf(serve): retain a reassignment capture at retry priority so it never displaces prompt or shared checkpoints|
|`72545ffdedf2`|`bb906f7aeea6`|perf(serve): capture an uncovered live frontier before its slot is reassigned|
|`15f27e88979f`|`71b3ffabc0f5`|fix(serve): do not reset an emptied idle slot again on every waiting admission pass|
|`033c28169515`|`4224e1c39c1b`|perf(qwen27b): map FP16 KV blocks of one shared pool as component-cache sessions write|
|`7daf08d4f6da`|`a4ce681a460a`|fix(qwen27b): count unused snapshot rows against the block-rounded KV rows|
|`11e583fff210`|`e604d732e874`|test(qwen): count FP16 KV in whole pool blocks in the storage accounting test|
|`ff5aeb7aef37`|`f8c9df1778d2`|perf(qwen27b): address FP16 attention rows through a block table over a plane-generic KV block pool|
|`14bcd820a980`|`235d1ed01805`|perf(qwen27b): share one prompt-pass scratch across component-cache sessions|
|`210c76337a5a`|`f0d3ff2dd653`|docs(cli): --cache-tail-checkpoints applies with paged KV only; test the cache option bounds|
|`5fc27fa6d789`|`9427815ab22a`|fix(cache): tail checkpoints only where snapshots share row blocks; restores keep records on resource failures|
|`f38d37e936fd`|`05b432ca3a09`|fix(cache): one shared prefix minimum, option bounds, tail checkpoints past existing stops|
|`e84d731c047a`|`676f834df8e0`|test(serving): back to the read-ahead 0/4 disk restores|
|`50b8fd4e2b79`|`97709389cb87`|fix(cache): read-ahead skips only blocks shared from residents, default 16|
|`30807fc19d07`|`c02e5206eca3`|feat(cache): --cache-disk-read-ahead N reads and checks row blocks ahead of their upload|
|`9a4b8bf2c859`|`ff42a3653380`|feat(cache): --cache-tail-checkpoints defaults to 3|
|`7b25ec7dffa9`|`271297122ecd`|fix(cache): tail checkpoints never cut a short piece from a full pass|
|`3d056d4d410e`|`6b5f757a71cf`|fix(cache): a checkpoint shared with peers drops tail checkpoints next to it|
|`8bdefd3cf487`|`9c810f0d3068`|fix(cache): count tail checkpoints back from where prefill stops|
|`45552058ce2d`|`741eda86df39`|fix(cache): a tail checkpoint never leaves a prompt pass below 1024 tokens|
|`d74f626dae77`|`f6980e0de652`|feat(cache): --cache-tail-checkpoints K, RAM checkpoints every 1024 tokens over the prompt end|
|`e5986004a164`|`09b0d05a409a`|feat(cache): --shared-prefix-min-tokens, default 128 (was a fixed 512)|
|`b15592c03922`|`877a68e410c9`|perf(qwen): touch the shared pass scratch at load, not in the first prompt pass|
|`46020c3a3429`|`19f7012ac3fc`|feat(qwen): one prompt-pass scratch shared by all sessions|
|`014d74e05821`|`70246aef1043`|feat(cache): --no-cache-resident-blocks turns the resident block index off|
|`14a577678bdc`|`6d668fd2efa6`|log(cache): time spent reading block files in a restore|
|`06eb0bff5896`|`8bca2e9b3e6c`|log(cache): count resident blocks a write-through deduplicated|
|`8af0574d77f5`|`e10b3d094c01`|feat(cache): index resident KV blocks by their block keys|
|`a5beb378d6d6`|`753e179bc218`|fix(cache): a request without prompt reuse is charged for every row block|
|`0296c09c4508`|`55dce2d08c8c`|perf(qwen): one KV block-table read per 16-key WMMA tile instead of per key|
|`64bf2aeb3df1`|`ec96f1144ba3`|perf(cache): paged sessions skip freezing a reused frontier right before a nearby stable boundary|
|`5cf3b473f3f2`|`62b660562f6f`|fix(cache): admission counts only blocks past a reused prefix, and synced block files leave the page cache|
|`e71947f5fe4c`|`c019979cfdf0`|feat(cache): checkpoint where the leading system messages end, an optional checkpoint grid, and optional idle record write-back|
|`2e0c9dc38f48`|`36d3f9dc76cb`|feat(cache): paged KV blocks on disk as hash-chained block files shared by every record of a prefix|
|`b6e0b2c7e737`|`5385c7193f94`|feat(qwen): paged KV sessions on one shared block pool, with block-sharing snapshots and copy-on-write|
|`9f27f5405a5c`|`20eef073d9e7`|test(serving): decode long enough for B's bounded prefill to join A, and count the retained prompt snapshot in the cleanup check|
|`3df7fb7bb5d3`|`8915bd3c3469`|test(serving): a disk restore may reuse the written-back reply frontier past the prompt|
|`c1a6933c8800`|`49ebf7cff526`|test(qwen): count FP16 KV in whole pool blocks in the storage accounting test|
|`c743e8399f26`|`cd0358e47638`|perf(qwen): address FP16 attention rows through a KV block table over a shared block pool|
|`be8186210a72`|`4d4f3b33cd54`|cache: write a released frontier only when an admission needs its seat; the scheduler requeues that admission instead of blocking|
|`edb48b112ef3`|`9590b014ccb3`|cache: write released frontiers to disk in place after 2 s idle and at shutdown; a resumed conversation cancels the write|
|`4e4b99a564b3`|`9eebc85bf6d8`|cache(qwen): stream a live session's compact form in place, sharing the snapshot segment builder|
|`a155bb0e148a`|`a2a22737e90a`|perf(cache): run bounce copies up to three pieces ahead of the writer|
|`9f66b879f748`|`712eb9faeb63`|cache: restore and verify disk images in bounded pieces; Qwen restores straight into the session through a bounce buffer|
|`d27de205d93d`|`02e687b50cc9`|cache(qwen): stream persistent snapshots through a pinned bounce buffer instead of a full-size host copy|
|`3e0297aa49eb`|`fd1273224364`|fix(gemma4): pass byte reserves to the backing pool after the upstream merge|
|`939d0375a73b`|`87f866a04eda`|Merge upstream/main (#650 recurrent buffers, #653, #524, #669) into ai-tools/cache-s4|
|`aa89d2a1928c`|`377f1eb498f9`|cache: report, not refuse, a retained snapshot budget below one max-context checkpoint|
|`0c8520b4ecc0`|`e1f8ec930af8`|test(cache): cover the max-context budget floor for RAM, disk and staging|
|`fa41fcffc380`|`b341601dd3cd`|cache: refuse explicit budgets below one max-context checkpoint (continuation snapshots)|

## 2026-10-25 저장소 이전과 이력 손실

- 작업 트리 `~/.cache/ai-tools/gufo/s4`는 `~/.cache/ai-tools/gufo/src` 저장소의 git worktree였다. 캐시 정리(사용자 지시) 때 `src`를 지워 git 데이터가 사라졌다. 에이전트가 두 경로를 별개 저장소로 잘못 판단한 탓이다.
- 다시 세운 저장소 `~/projects/gufo`는 포크 `namote/gufo`에서 받았다: `ai-tools/port-u4f` `0341dd88`, `ai-tools/port-tidy` `aeffca08`과 그 조상(이식 U0–U4f 전 커밋)은 그대로다.
- 잃은 것: 운영 계열(`cache-s4`, `paged-kv*`, `shared-prefill`, `upstream-0.21-merge`) 커밋 41개의 개별 이력. 위 표의 새 해시 중 저장소에 없는 것이 그 커밋들이다. 일부(7개)는 `storage/evaluations/gufo-eval-20261020/`·`test/experiments/*/patches/`에 format-patch 사본이 있다.
- 운영 최종 코드는 남아 있다: `ai-tools/production-210c7633` `8f75fdf7` = franzmoca `a8e40fd8` + ai.tools `infra/services/llm/gufo/ai-tools-210c7633.patch`(커밋 하나). 이미지 `ai-tools/gufo-eval:210c76337a5a` 안 소스와 `__pycache__` 외 동일함을 확인했고, 포크에도 push했다.
