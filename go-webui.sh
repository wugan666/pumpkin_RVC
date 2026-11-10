#!/usr/bin/env bash
# Start the WebUI. Uses .venv when it exists; extra arguments are passed on,
# e.g. ./go-webui.sh --noautoopen --port 7865
cd "$(dirname "$0")" || exit 1
PYTHON=python3
[ -x .venv/bin/python ] && PYTHON=.venv/bin/python
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" webui.py "$@"
