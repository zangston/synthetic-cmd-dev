#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <notebook.ipynb>"
    exit 1
fi

NOTEBOOK_PATH="$1"

if [ ! -f "$NOTEBOOK_PATH" ]; then
    echo "Error: notebook not found: $NOTEBOOK_PATH"
    exit 1
fi

NOTEBOOK_DIR="$(cd "$(dirname "$NOTEBOOK_PATH")" && pwd)"
NOTEBOOK_FILE="$(basename "$NOTEBOOK_PATH")"
NOTEBOOK_STEM="${NOTEBOOK_FILE%.ipynb}"

OUTPUT_FILE="${NOTEBOOK_STEM}_executed.ipynb"
LOG_FILE="${NOTEBOOK_STEM}_execution.log"
PID_FILE="${NOTEBOOK_STEM}_execution.pid"

cd "$NOTEBOOK_DIR"

nohup python -m jupyter nbconvert \
    --to notebook \
    --execute "$NOTEBOOK_FILE" \
    --output "$OUTPUT_FILE" \
    --ExecutePreprocessor.timeout=-1 \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

echo "Notebook started in the background."
echo "PID:             $PID"
echo "Input notebook:  $NOTEBOOK_DIR/$NOTEBOOK_FILE"
echo "Output notebook: $NOTEBOOK_DIR/$OUTPUT_FILE"
echo "Log file:        $NOTEBOOK_DIR/$LOG_FILE"
echo "PID file:        $NOTEBOOK_DIR/$PID_FILE"