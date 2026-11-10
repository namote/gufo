#!/usr/bin/env bash
# Serve the deployed Flash-Next c03 EXL3 pack with Kyojin v1.8.6 for quality
# comparison against Gufo, on Gufo's endpoint (0.0.0.0:18740, model id
# gufo-flash-next). Stop Gufo first (`llm stop`); restore with
# `kyojin-serve.sh stop && llm start`.
# Usage: tools/serving/kyojin/kyojin-serve.sh start|stop|status|logs
# Runtime: tools/exl3_quant/kyojin/fetch_kyojin.sh + build_kyojin.sh into $RT,
# then `pin.py patch` (serve.py --max-context).
# Defaults follow tools/serving/flash_next/gufo-server.py: 262144-token pool,
# 2 sessions, fp16 KV, thinking on at medium, temperature 0.9, top-k 20,
# top-p 0.95, min-p 0. Kyojin has no Han scale. Speculation is Kyojin's
# shipped `mix` rule (MTP + n-gram); KYOJIN_DRAFT=off serves plain decode.
set -euo pipefail
RT=${KYOJIN_RT:-/home/m2on/.local/share/gufo/kyojin-runtime-20261110}
MODEL=${KYOJIN_MODEL:-/home/m2on/projects/models/llm/flash-next-exl3-gufo-c03}
IMAGE=${KYOJIN_IMAGE:-ai-tools/gufo-eval:native-exl3-vision-rocm101}
NAME=kyojin-fnx
PORT=18740

case "${1:-}" in
start)
  if docker ps -q --filter "name=^gufo-" | grep -q .; then
    echo "Gufo is running; run 'llm stop' first" >&2; exit 1
  fi
  if ss -ltnH "sport = :$PORT" | grep -q .; then echo "port $PORT is in use" >&2; exit 1; fi
  free_gib=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
  [ "$free_gib" -ge 100 ] || { echo "disk free ${free_gib}G < 100G" >&2; exit 1; }
  mkdir -p "$RT/home" "$RT/slots"
  groups=()
  for g in $(stat -c %g /dev/kfd /dev/dri/renderD* | sort -u); do groups+=(--group-add "$g"); done
  docker run -d --rm --name "$NAME" --pull never --network host --user 1000:1000 \
    --memory 110g --memory-swap 110g --device /dev/kfd --device /dev/dri --ipc host "${groups[@]}" \
    --tmpfs /tmp:rw,size=2g,mode=1777 \
    --mount "type=bind,src=$RT,dst=/runtime" \
    --mount "type=bind,src=$MODEL,dst=/models/fnx,readonly" \
    -e HOME=/runtime/home -e PYTHONUNBUFFERED=1 \
    -e PYTHONPATH=/runtime/site -e EXL3_ROCM_SDK=/runtime/site/_rocm_sdk_devel \
    -e EXL3_ROOT=/runtime/source -e EXL3_REPO=/runtime/source -e EXL3_VENV=/opt/venv \
    --entrypoint /bin/bash "$IMAGE" -c "
set -euo pipefail
cd /runtime/source
source tools/strix_halo/env.sh
export PATH=/runtime/site/bin:\$PATH
exec python -u tools/qwen/serve.py --host 0.0.0.0 --port $PORT --model /models/fnx \
  --model-id gufo-flash-next -c 262144 --sessions 2 --draft-policy ${KYOJIN_DRAFT:-mix} \
  --default-temperature 0.9 --default-top-k 20 --default-top-p 0.95 --default-min-p 0 \
  --default-reasoning-effort medium --slot-save-path /runtime/slots"
  echo "started $NAME; follow with: $0 logs" ;;
stop) docker stop "$NAME" ;;
status) docker ps --filter "name=^$NAME\$" --format '{{.Names}} {{.Status}}'
        curl -s --max-time 5 "http://127.0.0.1:$PORT/health" || echo "not ready" ;;
logs) docker logs -f "$NAME" ;;
*) sed -n '2,12p' "$0" >&2; exit 2 ;;
esac
