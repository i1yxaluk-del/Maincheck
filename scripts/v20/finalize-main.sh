#!/usr/bin/env bash
set -euo pipefail
ROOT=${MAINCHK_ROOT:-/home/service/llama}; LOCAL="$ROOT/server/local"; VENV="$LOCAL/venv"; LOG=${V20_FINALIZE_LOG:-/var/log/maincheck-v20-finalize.log}
if [[ ${EUID:-$(id -u)} -ne 0 ]]; then echo 'run with sudo' >&2; exit 1; fi
exec > >(tee -a "$LOG") 2>&1
log(){ printf '%s %s\n' "$(date -Is)" "$*"; }
log 'V20 FINALIZE BEGIN selected=main'
BEFORE=$(du -sb "$ROOT" 2>/dev/null | awk '{print $1}' || echo 0)
for unit in v20-openvino.service v20-llama-json.service v20-llama-backend.service; do systemctl disable --now "$unit" 2>/dev/null || true; done
while read -r unit; do [[ -n "$unit" ]] && systemctl stop "$unit" 2>/dev/null || true; done < <(systemctl list-units --all --plain --no-legend 'ai-suggester-v20-install-*' 2>/dev/null | awk '{print $1}')
rm -rf \
  "$LOCAL/venv-openvino" \
  "$LOCAL/venv-llama-json" \
  "$ROOT/models/v20" \
  "$ROOT/vendor/llama.cpp"
rm -f /run/maincheck-v20-ready /run/maincheck-v20-install.status
# Earlier experiments may have placed OpenVINO/CUDA packages in the shared CPU venv.
# Replace Torch with a CPU wheel first; only then remove orphan accelerator packages.
if [[ ${V20_PURGE_SHARED_EXPERIMENT_DEPS:-true} == true && -x "$VENV/bin/pip" ]]; then
  log 'V20 FINALIZE shared-venv CPU cleanup'
  if "$VENV/bin/pip" install --disable-pip-version-check --force-reinstall --no-deps --index-url https://download.pytorch.org/whl/cpu 'torch>=2.5,<3'; then
    mapfile -t extras < <("$VENV/bin/pip" list --format=freeze | sed 's/==.*//' | grep -E '^(nvidia-|cuda-|triton$|optimum-intel$|optimum-onnx$|openvino$|openvino-|nncf$)' || true)
    ((${#extras[@]}==0)) || "$VENV/bin/pip" uninstall -y "${extras[@]}"
  else
    log 'WARNING: CPU Torch reinstall failed; shared packages left untouched'
  fi
fi
"$ROOT/scripts/v20/install-worker.sh" main
# install-worker deploys all source unit files for profile switching; remove the
# discarded experimental units after the selected service is healthy.
for unit in v20-openvino.service v20-llama-json.service v20-llama-backend.service; do systemctl disable --now "$unit" 2>/dev/null || true; rm -f "/etc/systemd/system/$unit"; done
systemctl daemon-reload; systemctl reset-failed
HEALTH=$(curl --max-time 10 -fsS http://127.0.0.1:8000/health)
AFTER=$(du -sb "$ROOT" 2>/dev/null | awk '{print $1}' || echo 0); FREED=$(( BEFORE>AFTER ? BEFORE-AFTER : 0 ))
log "V20 FINALIZED profile=main freed_bytes=$FREED health=$HEALTH"
log 'Rollback remains: systemctl disable --now v20-main.service && systemctl enable --now ai-suggester.service'
