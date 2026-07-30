#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <notebook.ipynb | notebook.py>"
    exit 1
fi

INPUT_PATH="$1"

if [ ! -f "$INPUT_PATH" ]; then
    echo "Error: file not found: $INPUT_PATH"
    exit 1
fi

INPUT_DIR="$(cd "$(dirname "$INPUT_PATH")" && pwd)"
INPUT_FILE="$(basename "$INPUT_PATH")"

cd "$INPUT_DIR"

EXT="${INPUT_FILE##*.}"
STEM="${INPUT_FILE%.*}"

case "$EXT" in
    ipynb)
        NOTEBOOK_FILE="$INPUT_FILE"
        TEMP_NOTEBOOK=""
        ;;
    py)
        if ! command -v jupytext >/dev/null 2>&1; then
            echo "Error: jupytext is not installed."
            echo "Install with:"
            echo "    pip install jupytext"
            exit 1
        fi

        TEMP_NOTEBOOK="${STEM}_temp.ipynb"

        echo "Converting Python script to notebook..."
        jupytext --to notebook "$INPUT_FILE" -o "$TEMP_NOTEBOOK"

        NOTEBOOK_FILE="$TEMP_NOTEBOOK"
        ;;
    *)
        echo "Error: supported file types are .ipynb and .py"
        exit 1
        ;;
esac

OUTPUT_NOTEBOOK="${STEM}_executed.ipynb"
LOG_FILE="${STEM}_execution.log"
PID_FILE="${STEM}_execution.pid"

cleanup() {
    if [[ -n "${TEMP_NOTEBOOK}" && -f "${TEMP_NOTEBOOK}" ]]; then
        rm -f "${TEMP_NOTEBOOK}"
    fi
}

nohup bash -c "
trap 'rm -f \"$TEMP_NOTEBOOK\"' EXIT

python -m jupyter nbconvert \
    --to notebook \
    --execute \"$NOTEBOOK_FILE\" \
    --output \"$OUTPUT_NOTEBOOK\" \
    --ExecutePreprocessor.timeout=-1
" > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

echo "Background execution started."
echo
echo "Input file:        $INPUT_DIR/$INPUT_FILE"
echo "Executed notebook: $INPUT_DIR/$OUTPUT_NOTEBOOK"
echo "Log file:          $INPUT_DIR/$LOG_FILE"
echo "PID file:          $INPUT_DIR/$PID_FILE"
echo "PID:               $PID"

if [[ "$EXT" == "py" ]]; then
    echo
    echo "Source type: Python (# %% cells)"
    echo "Temporary notebook: $TEMP_NOTEBOOK"
fi