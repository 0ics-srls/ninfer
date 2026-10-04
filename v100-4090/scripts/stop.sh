#!/bin/bash
# stop.sh — stops the engine started by start-tp.sh (the container is --rm: it disappears; logs stay in LOGS).
docker stop ninfer-v100-4090 >/dev/null 2>&1 && echo "stopped" || echo "not running"
