#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

python3 -m venv venv 2>/dev/null || true
venv/bin/python -m pip install --upgrade pip
venv/bin/pip install -r requirements.txt

if command -v systemctl >/dev/null 2>&1; then
  sudo cp ai-suggester.service /etc/systemd/system/ai-suggester.service
  sudo systemctl daemon-reload
  sudo systemctl enable ai-suggester.service
fi

if command -v java >/dev/null 2>&1 && [[ "${INSTALL_LANGUAGETOOL:-true}" == "true" ]]; then
  echo "LanguageTool is enabled through LANGUAGETOOL_ENABLED and must be installed as a local Java service."
  echo "Set LANGUAGETOOL_URL to its local /v2 endpoint before restarting ai-suggester."
fi

echo "Installed. Configure .env, then run: sudo systemctl restart ai-suggester.service"
