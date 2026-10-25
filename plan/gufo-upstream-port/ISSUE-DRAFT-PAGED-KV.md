# 이슈 초안: paged live KV 방향 문의 (올리지 않음)

상위 문서: [이식 계획](PLAN.md) U6·D3, [PR 분할 계획](PR-PLAN.md).

사용자 결정(2026-10-25): paged KV는 PR보다 이슈로 방향부터 묻는다. 초안만 두고 올리지 않는다. 공개 계정·이메일(D2)이 정해지기 전에는 이슈·PR·push를 하지 않는다.

대상 저장소: `gufo-org/gufo`. 아래 본문은 원본 언어(영어)로 쓴다. 수치는 우리 장비 한 대(gfx1151) 측정이며, 원본 CI 환경(Nix GCC 15.3)에서는 아직 확인하지 않았다.

---

**Title:** Paged live KV for Qwen 27B component-cache sessions: a direction check before PRs

**Body:**

We run gufo on a Strix Halo (gfx1151, unified memory) with Qwen3.8-27B (GGUF Q4_K_M, DFlash2 drafts), 256K context and 4 sessions. We have a working series on top of `main` (82c62b40) and would like to know whether it fits the direction of the KV-cache RFC follow-ups before opening PRs.

What it does

- One shared device pool of 256-token KV blocks (FP16 attention rows) instead of a full-context KV allocation per session. Attention kernels read rows through a per-session block table (decode, verify, tiled and WMMA prefill).
- Sessions map blocks as they write. Admission reserves the blocks a request needs (prompt plus `max_tokens`, counting blocks a continued session already maps) and waits, or reclaims idle rows, when the pool is short.
- Component-cache checkpoints keep KV chunks as block references (reference-counted) instead of RAM copies. A write into a shared block copies that block first. A restore into another slot maps the retained blocks.
- With no option the pool holds every session's full context, the same device memory as today. `--kv-pool-bytes N` sizes it explicitly. Other runners refuse a nonzero value.

Why

- Checkpoint retention no longer copies KV into RAM. A RAM checkpoint then costs only its private recurrent state and partial tails (about 250 MB at 8K tokens here, instead of the full KV). A 2 GiB RAM budget keeps the same time to first token as larger budgets in our runs.
- A pool smaller than sessions × context overcommits safely (admission waits). In our measurement, 64K × 8 sessions ran on a 24 GiB pool.

Measured (gfx1151, same model and prompts, texts identical to the unpaged build in every case)

- Decode and prefill throughput are within noise (bench pp2048 and tg128 within ±1 %).
- RAM restore of the next turn dropped from 0.50 s to 0.28 s. Restoring into another slot maps blocks instead of copying.
- Batch-shared prompt-pass scratch (a separate, smaller change) frees 6.4 GiB at 16K × 4 and 14.7 GiB at 64K × 8 at startup.

Questions

1. Is a shared block pool for the Qwen 27B component-cache path acceptable, or should paging first become a model-agnostic cache facility (Flash-Next and DeepSeek V4 Flash keep their own layouts today)?
2. Is a 256-token block (one component-cache chunk) the right granularity, or would you prefer the block size to be independent of the chunk size?
3. The series is about 30 commits. We propose small PRs in this order: shared pass scratch → block table kernels → shared pool and admission → checkpoint block references → reassignment capture and idle write-back. Would you take them in that order?

Independent fixes that do not depend on paging (a `ShareCheckpoint` boundary case, restore and write-back logging) would come as separate PRs first.

---

## 올리기 전에 할 일

- D2(공개 계정·이메일) 결정.
- 수치를 원본 기준(U0) 창과 같은 구성으로 다시 쟀는지 확인. 위 수치는 우리 운영 구성(256K × 4) 기준이다.
- 원본 RFC 후속 과제 문서의 현재 상태를 다시 읽고 질문 1을 맞춘다.
