#!/usr/bin/env bash
set -euo pipefail
PROFILE=${1:?profile}; ROOT=${MAINCHK_ROOT:-/home/service/llama}; LOCAL="$ROOT/server/local"; VENV="$LOCAL/venv"; ENV="$LOCAL/.env.v20"; MODEL_DIR="$ROOT/models/v20"; LLAMA_DIR="$ROOT/vendor/llama.cpp"
log(){ printf '%s %s\n' "$(date -Is)" "$*"; }
trap 'log "V20 INSTALL FAILED profile=$PROFILE line=$LINENO"' ERR
log "V20 INSTALL BEGIN profile=$PROFILE"
install -d -o service -g service "$MODEL_DIR" "$ROOT/results" "$ROOT/vendor"
for u in ai-suggester.service v20-main.service v20-openvino.service v20-llama-json.service v20-llama-backend.service; do systemctl disable --now "$u" 2>/dev/null || true; done
if [[ "$PROFILE" != main ]]; then systemctl stop ollama.service 2>/dev/null || true; fi
"$VENV/bin/pip" install --disable-pip-version-check -r "$LOCAL/requirements.txt"
"$VENV/bin/pip" install --disable-pip-version-check python-multipart
if [[ "$PROFILE" == openvino ]]; then "$VENV/bin/pip" install --disable-pip-version-check 'openvino>=2025.4,<2027' 'optimum-intel>=1.24' sentencepiece; fi
if [[ "$PROFILE" == llama-json && ! -x "$LLAMA_DIR/build/bin/llama-server" ]]; then
  command -v git >/dev/null; command -v cmake >/dev/null; command -v g++ >/dev/null
  [[ -d "$LLAMA_DIR/.git" ]] || sudo -u service git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
  sudo -u service cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DGGML_NATIVE=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
  sudo -u service cmake --build "$LLAMA_DIR/build" --config Release -j "$(nproc)"
fi
TMP=$(mktemp "${ENV}.XXXX")
cat >"$TMP" <<EOF
V20_PROFILE=$PROFILE
LLM_PRESET=Z
RAG_ENABLED=true
RAG_STORE_DIR=$ROOT/RAG/state
RAG_DOCUMENTS_DIR=$ROOT/RAG/documents
REASONING_MODE=coverage
REASONING_TOTAL_TIMEOUT=20
OLLAMA_URL=http://127.0.0.1:11434
V20_OPENVINO_MODEL=${V20_OPENVINO_MODEL:-ai-forever/sage-fredt5-large}
V20_LLAMA_URL=http://127.0.0.1:8091
V20_LLAMA_MODEL_NAME=local-gec
V20_LLAMA_TIMEOUT=25
V20_LLAMA_THREADS=${V20_LLAMA_THREADS:-$(nproc)}
V20_LLAMACPP_MODEL=${V20_LLAMACPP_MODEL:-$MODEL_DIR/qwen3-8b-q4_k_m.gguf}
EOF
install -o service -g service -m 0640 "$TMP" "$ENV"; rm -f "$TMP"
for f in "$LOCAL"/v20-*.service; do install -m 0644 "$f" /etc/systemd/system/; done
systemctl daemon-reload
case "$PROFILE" in
 main) systemctl enable --now ollama.service 2>/dev/null || true; systemctl enable --now v20-main.service; SERVICE=v20-main.service;;
 openvino) systemctl enable --now v20-openvino.service; SERVICE=v20-openvino.service;;
 llama-json) [[ -s "${V20_LLAMACPP_MODEL:-$MODEL_DIR/qwen3-8b-q4_k_m.gguf}" ]] || { log "missing GGUF: set V20_LLAMACPP_MODEL"; exit 3; }; systemctl enable --now v20-llama-backend.service v20-llama-json.service; SERVICE=v20-llama-json.service;;
esac
for _ in $(seq 1 180); do if curl -fsS http://127.0.0.1:8000/health >/dev/null; then log "V20 READY profile=$PROFILE service=$SERVICE"; exit 0; fi; sleep 5; done
log "health timeout profile=$PROFILE"; systemctl status "$SERVICE" --no-pager || true; exit 4
