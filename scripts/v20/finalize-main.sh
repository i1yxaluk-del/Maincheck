#!/usr/bin/env bash
set -euo pipefail
ROOT=${MAINCHK_ROOT:-/home/service/llama}; LOCAL="$ROOT/server/local"; VENV="$LOCAL/venv"; ENV="$LOCAL/.env.v20"; LOG=${V20_FINALIZE_LOG:-/var/log/maincheck-v20-finalize.log}; UNIT=/etc/systemd/system/ai-suggester.service; BACKUP=/etc/systemd/system/ai-suggester.service.pre-v20
if [[ ${EUID:-$(id -u)} -ne 0 ]]; then echo 'run with sudo' >&2; exit 1; fi
exec > >(tee -a "$LOG") 2>&1
log(){ printf '%s %s\n' "$(date -Is)" "$*"; }
fail(){ code=$?; log "V20 FINALIZE FAILED line=$1 code=$code"; systemctl status ai-suggester.service ollama.service --no-pager || true; exit "$code"; }; trap 'fail $LINENO' ERR
log 'V20 FINALIZE BEGIN target=ai-suggester.service stack=Z'
BEFORE=$(du -sb "$ROOT" 2>/dev/null | awk '{print $1}' || echo 0)
for unit in v20-main.service v20-openvino.service v20-llama-json.service v20-llama-backend.service ai-suggester.service; do systemctl disable --now "$unit" 2>/dev/null || true; done
while read -r unit; do [[ -n "$unit" ]] && systemctl stop "$unit" 2>/dev/null || true; done < <(systemctl list-units --all --plain --no-legend 'ai-suggester-v20-install-*' 2>/dev/null | awk '{print $1}')
[[ -e "$UNIT" && ! -e "$BACKUP" ]] && cp -a "$UNIT" "$BACKUP" || true
rm -rf "$LOCAL/venv-openvino" "$LOCAL/venv-llama-json" "$ROOT/models/v20" "$ROOT/vendor/llama.cpp"
rm -f /run/maincheck-v20-ready /run/maincheck-v20-install.status
if [[ ${V20_PURGE_SHARED_EXPERIMENT_DEPS:-true} == true && -x "$VENV/bin/pip" ]]; then
 log 'V20 FINALIZE shared-venv CPU cleanup'
 if "$VENV/bin/pip" install --disable-pip-version-check --force-reinstall --no-deps --index-url https://download.pytorch.org/whl/cpu 'torch>=2.5,<3'; then
  mapfile -t extras < <("$VENV/bin/pip" list --format=freeze | sed 's/==.*//' | grep -E '^(nvidia-|cuda-|triton$|optimum-intel$|optimum-onnx$|openvino$|openvino-|nncf$)' || true)
  ((${#extras[@]}==0)) || "$VENV/bin/pip" uninstall -y "${extras[@]}"
 else log 'WARNING: CPU Torch reinstall failed; shared packages left untouched'; fi
fi
"$VENV/bin/pip" install --disable-pip-version-check -r "$LOCAL/requirements.txt" python-multipart
TMP=$(mktemp "${ENV}.XXXX"); cat >"$TMP" <<EOF
V20_PROFILE=main
LLM_PRESET=Z
RAG_ENABLED=true
RAG_STORE_DIR=$ROOT/RAG/state
RAG_DOCUMENTS_DIR=$ROOT/RAG/documents
REASONING_MODE=coverage
V20_MAIN_REASONING_TIMEOUT=12
REASONING_TOTAL_TIMEOUT=12
OLLAMA_URL=http://127.0.0.1:11434
DECISION_MIN_CONFIDENCE=0.65
DECISION_MAX_CHANGES=8
ARBITER_SOLO_PENALTY=0.35
EOF
install -o service -g service -m 0640 "$TMP" "$ENV"; rm -f "$TMP"
install -m 0644 "$LOCAL/ai-suggester.service" "$UNIT"
for unit in v20-main.service v20-openvino.service v20-llama-json.service v20-llama-backend.service; do rm -f "/etc/systemd/system/$unit" "/etc/systemd/system/multi-user.target.wants/$unit"; done
systemctl daemon-reload; systemctl reset-failed
systemctl enable --now ollama.service
systemctl enable --now ai-suggester.service
HEALTH=''
for _ in $(seq 1 60); do HEALTH=$(curl --max-time 10 -sS http://127.0.0.1:8000/health 2>/dev/null || true); [[ "$HEALTH" == *'stack=Z'* && "$HEALTH" != DEGRADED* ]] && break; sleep 5; done
[[ "$HEALTH" == *'stack=Z'* && "$HEALTH" != DEGRADED* ]] || { log "unexpected health=$HEALTH"; false; }
AFTER=$(du -sb "$ROOT" 2>/dev/null | awk '{print $1}' || echo 0); FREED=$(( BEFORE>AFTER ? BEFORE-AFTER : 0 ))
log "V20 FINALIZED service=ai-suggester.service stack=Z freed_bytes=$FREED health=$HEALTH"
log "Rollback: cp -a $BACKUP $UNIT && systemctl daemon-reload && systemctl restart ai-suggester.service"
