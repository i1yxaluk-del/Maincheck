#!/bin/sh
# Start a private local LanguageTool service for punctuation and typos.
set -eu
IMAGE=${LANGUAGETOOL_IMAGE:-erikvl87/languagetool:latest}
if command -v podman >/dev/null 2>&1; then
  RUNTIME=podman
elif command -v docker >/dev/null 2>&1; then
  RUNTIME=docker
else
  echo "ERROR: install podman or docker, then rerun this script" >&2
  exit 1
fi
$RUNTIME rm -f languagetool >/dev/null 2>&1 || true
$RUNTIME run -d --name languagetool --restart=always \
  -p 127.0.0.1:8081:8010 -e Java_Xmx=2g "$IMAGE"
echo "Waiting for LanguageTool..."
i=0
until curl -fsS http://127.0.0.1:8081/v2/languages >/dev/null 2>&1; do
  i=$((i + 1)); [ "$i" -lt 60 ] || { echo "LanguageTool did not start" >&2; exit 1; }
  sleep 2
done
echo "LanguageTool ready on 127.0.0.1:8081"
