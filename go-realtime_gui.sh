#!/usr/bin/env bash
# Start the realtime voice changer. Uses .venv when it exists.
cd "$(dirname "$0")" || exit 1
PYTHON=python3
[ -x .venv/bin/python ] && PYTHON=.venv/bin/python
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" realtime_gui.py "$@"
