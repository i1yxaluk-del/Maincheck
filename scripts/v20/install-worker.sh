#!/usr/bin/env bash
set -euo pipefail
PROFILE=${1:?profile}; ROOT=${MAINCHK_ROOT:-/home/service/llama}; LOCAL="$ROOT/server/local"; VENV="$LOCAL/venv"; OV_VENV="$LOCAL/venv-openvino"; ENV="$LOCAL/.env.v20"; MODEL_DIR="$ROOT/models/v20"; LLAMA_DIR="$ROOT/vendor/llama.cpp"; LOG=${V20_INSTALL_LOG:-/var/log/ai-suggester-v20-install.log}; STATUS=/run/maincheck-v20-install.status; READY=/run/maincheck-v20-ready
exec > >(tee -a "$LOG") 2>&1
log(){ printf '%s %s\n' "$(date -Is)" "$*"; }
fail(){ code=$?; printf 'failed profile=%s line=%s code=%s\n' "$PROFILE" "$1" "$code" >"$STATUS"; log "V20 INSTALL FAILED profile=$PROFILE line=$1 code=$code"; exit "$code"; }; trap 'fail $LINENO' ERR
rm -f "$READY"; printf 'running profile=%s\n' "$PROFILE" >"$STATUS"; log "V20 INSTALL BEGIN profile=$PROFILE"
install -d -o service -g service "$MODEL_DIR" "$ROOT/results" "$ROOT/vendor"
for u in ai-suggester.service v20-main.service v20-openvino.service v20-llama-json.service v20-llama-backend.service; do systemctl disable --now "$u" 2>/dev/null || true; done
if [[ "$PROFILE" != main ]]; then systemctl stop ollama.service 2>/dev/null || true; fi
pkg_build(){ if command -v dnf >/dev/null; then dnf install -y git cmake gcc-c++ openblas-devel curl; elif command -v apt-get >/dev/null; then apt-get update; DEBIAN_FRONTEND=noninteractive apt-get install -y git cmake g++ libopenblas-dev curl; else log 'no supported package manager'; return 1; fi; }
pkg_venv(){ if command -v dnf >/dev/null; then dnf install -y python3; elif command -v apt-get >/dev/null; then apt-get update; DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv; else return 1; fi; }
OPENVINO_RUNTIME=${V20_OPENVINO_MODEL:-ai-forever/sage-fredt5-large}
if [[ "$PROFILE" == openvino ]]; then
  if [[ ! -x "$OV_VENV/bin/python" ]]; then python3 -m venv "$OV_VENV" || { pkg_venv; python3 -m venv "$OV_VENV"; }; chown -R service:service "$OV_VENV"; fi
  "$OV_VENV/bin/pip" install --disable-pip-version-check -r "$LOCAL/requirements-v20-openvino.txt"
  OV_SOURCE=${V20_OPENVINO_MODEL:-ai-forever/sage-fredt5-large}; OV_DIR="$MODEL_DIR/openvino-sage-fredt5-large"
  if [[ ! -s "$OV_DIR/openvino_encoder_model.xml" || ! -s "$OV_DIR/openvino_decoder_model.xml" ]]; then
    TMP_OV="$OV_DIR.tmp"; rm -rf "$TMP_OV"; install -d -o service -g service "$TMP_OV" "$MODEL_DIR/hf-cache"
    log "V20 OPENVINO EXPORT source=$OV_SOURCE target=$OV_DIR"
    runuser -u service -- env HF_HOME="$MODEL_DIR/hf-cache" "$OV_VENV/bin/optimum-cli" export openvino --model "$OV_SOURCE" --task text2text-generation-with-past "$TMP_OV"
    rm -rf "$OV_DIR"; mv "$TMP_OV" "$OV_DIR"; chown -R service:service "$OV_DIR"
  fi
  OPENVINO_RUNTIME=$OV_DIR
else
  "$VENV/bin/pip" install --disable-pip-version-check -r "$LOCAL/requirements.txt" python-multipart
fi
if [[ "$PROFILE" == llama-json ]]; then
  if ! command -v git >/dev/null || ! command -v cmake >/dev/null || ! command -v g++ >/dev/null; then pkg_build; fi
  if [[ ! -x "$LLAMA_DIR/build/bin/llama-server" ]]; then
    [[ -d "$LLAMA_DIR/.git" ]] || runuser -u service -- git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
    runuser -u service -- cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DGGML_NATIVE=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
    runuser -u service -- cmake --build "$LLAMA_DIR/build" --config Release -j "$(nproc)"
  fi
  MODEL=${V20_LLAMACPP_MODEL:-$MODEL_DIR/qwen3-8b-q4_k_m.gguf}; URL=${V20_LLAMACPP_MODEL_URL:-https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf}
  if [[ ! -s "$MODEL" ]]; then log "downloading GGUF to $MODEL"; curl -fL --retry 5 -C - "$URL" -o "$MODEL"; chown service:service "$MODEL"; fi
fi
TMP=$(mktemp "${ENV}.XXXX"); cat >"$TMP" <<EOF
V20_PROFILE=$PROFILE
LLM_PRESET=Z
RAG_ENABLED=true
RAG_STORE_DIR=$ROOT/RAG/state
RAG_DOCUMENTS_DIR=$ROOT/RAG/documents
REASONING_MODE=coverage
REASONING_TOTAL_TIMEOUT=20
OLLAMA_URL=http://127.0.0.1:11434
V20_OPENVINO_MODEL=$OPENVINO_RUNTIME
V20_LLAMA_URL=http://127.0.0.1:8091
V20_LLAMA_MODEL_NAME=local-gec
V20_LLAMA_TIMEOUT=25
V20_LLAMA_THREADS=${V20_LLAMA_THREADS:-$(nproc)}
V20_LLAMACPP_MODEL=${V20_LLAMACPP_MODEL:-$MODEL_DIR/qwen3-8b-q4_k_m.gguf}
EOF
install -o service -g service -m 0640 "$TMP" "$ENV"; rm -f "$TMP"
for f in "$LOCAL"/v20-*.service; do install -m 0644 "$f" /etc/systemd/system/; done; systemctl daemon-reload
case "$PROFILE" in main) systemctl enable --now ollama.service 2>/dev/null || true; systemctl enable --now v20-main.service; SERVICE=v20-main.service;; openvino) systemctl enable --now v20-openvino.service; SERVICE=v20-openvino.service;; llama-json) systemctl enable --now v20-llama-backend.service v20-llama-json.service; SERVICE=v20-llama-json.service;; esac
for i in $(seq 1 180); do
  REPLY=$(curl --max-time 30 -sS -w $'\n%{http_code}' http://127.0.0.1:8000/health 2>&1 || true); HTTP=${REPLY##*$'\n'}; BODY=${REPLY%$'\n'*}
  if [[ "$HTTP" == 200 ]]; then printf '%s\n' "$PROFILE" >"$READY"; printf 'ready profile=%s service=%s\n' "$PROFILE" "$SERVICE" >"$STATUS"; log "V20 READY profile=$PROFILE service=$SERVICE"; exit 0; fi
  if [[ "$PROFILE" == openvino && "$HTTP" == 503 ]]; then log "V20 OPENVINO HEALTH FAILED body=$BODY"; false; fi
  (( i % 6 == 0 )) && log "V20 HEALTH WAIT profile=$PROFILE http=$HTTP body=$BODY"
  sleep 5
done
log "health timeout profile=$PROFILE"; systemctl status "$SERVICE" --no-pager || true; exit 4
