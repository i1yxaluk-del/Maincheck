#!/usr/bin/env bash
set -euo pipefail
PROFILE=${1:-main}; ROOT=${MAINCHK_ROOT:-/home/service/llama}
case "$PROFILE" in main|openvino|llama-json) ;; *) echo "usage: $0 main|openvino|llama-json" >&2; exit 2;; esac
LOG=${V20_INSTALL_LOG:-/var/log/ai-suggester-v20-install.log}
UNIT="ai-suggester-v20-install-${PROFILE}-$(date +%Y%m%d%H%M%S)-$$"
if [[ ${EUID:-$(id -u)} -ne 0 ]]; then echo "run with sudo" >&2; exit 1; fi
systemd-run --unit="$UNIT" --collect --property=Type=exec /bin/bash "$ROOT/scripts/v20/install-worker.sh" "$PROFILE"
echo "V20 INSTALL STARTED unit=$UNIT log=$LOG"
echo "journalctl -fu $UNIT"
