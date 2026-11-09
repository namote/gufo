#!/usr/bin/env bash
# Host preflight for gfx1151 timing windows (2026-11-10, docs/GFX1151-KNOWHOW.md).
# Fixes and records the host state that swung U6/U7 timings by 10-40%:
# page cache, fragmentation, compaction sysctl, THP, disk space, other jobs.
# Usage: tools/bench/host_preflight.sh <out-dir> [min-free-GiB]
# Exits non-zero (and prints why) when the host is not fit for timing.
set -u
out=${1:?usage: host_preflight.sh <out-dir> [min-free-GiB]}
min_free_gib=${2:-150}
mkdir -p "$out"
log="$out/preflight-$(date +%Y%m%dT%H%M%S).txt"
fail=0
note() { echo "$*" | tee -a "$log"; }
bad() { note "FAIL: $*"; fail=1; }

note "date $(date -Is)"
note "uptime $(uptime)"
note "kernel $(uname -r)"
note "linux-firmware $(dpkg-query -W -f='${Version}' linux-firmware 2>/dev/null)"
for p in cwsr_enable queue_preemption_timeout_ms; do
  note "amdgpu.$p $(cat /sys/module/amdgpu/parameters/$p 2>/dev/null)"
done

# Compaction and THP must be at the measured-good defaults.
pro=$(cat /proc/sys/vm/compaction_proactiveness)
note "vm.compaction_proactiveness $pro"
[ "$pro" = 20 ] || bad "compaction_proactiveness is $pro, expected 20 (50 cost MTP ~18%, U7)"
note "thp.enabled $(cat /sys/kernel/mm/transparent_hugepage/enabled)"
note "thp.defrag $(cat /sys/kernel/mm/transparent_hugepage/defrag)"

# No other GPU users or heavy CPU jobs (builds contaminate shared LPDDR timing).
gpu_ctr=$(docker ps --format '{{.Names}} {{.Image}}' | grep -Ei 'gufo|rocm|llama|vllm' || true)
[ -z "$gpu_ctr" ] || bad "GPU containers running: $gpu_ctr"
# %CPU is a lifetime average, so a just-started process (this ps, awk) reads
# near 100; only count processes older than 5 s.
heavy=$(ps -eo pcpu=,etimes=,comm= --sort=-pcpu | awk '$1 > 50 && $2 >= 5 {print $1, $3}' | head -5)
[ -z "$heavy" ] || bad "busy processes: $heavy"
note "load $(cut -d' ' -f1-3 /proc/loadavg)"

# Disk: enough free space for disk-cache suites, then trim.
free_gib=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
note "disk_free_gib $free_gib"
[ "$free_gib" -ge "$min_free_gib" ] || bad "disk free ${free_gib}G < ${min_free_gib}G"
sudo -n fstrim -v / 2>&1 | tee -a "$log" >/dev/null || note "fstrim unavailable"

# Page cache and fragmentation.
note "buddyinfo before:"; cat /proc/buddyinfo >>"$log"
sync
sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory' \
  || bad "drop_caches/compact_memory failed (sudo -n)"
note "buddyinfo after:"; cat /proc/buddyinfo >>"$log"
order9=$(awk '/Normal/ {for (i=14; i<=NF; i++) s+=$i} END {print s+0}' /proc/buddyinfo)
note "free_blocks_order_ge9 $order9"
grep -E 'MemFree|^Cached|Dirty' /proc/meminfo >>"$log"

if [ "$fail" = 0 ]; then note "PREFLIGHT OK"; else note "PREFLIGHT FAILED"; fi
exit "$fail"
