#!/usr/bin/env bash

# Deletes nested:
#   .ipynb_checkpoints
#   isochrones
#   __pycache__
#
# Default: preview only
# Use -f to actually delete

set -e

TARGET_DIR="${1:-.}"
MODE="${2:-preview}"

echo "Searching in: $TARGET_DIR"
echo

matches=$(find "$TARGET_DIR" -type d \( \
    -name ".ipynb_checkpoints" -o \
    -name "isochrones" -o \
    -name "__pycache__" \
\))

if [ -z "$matches" ]; then
    echo "No matching directories found."
    exit 0
fi

echo "Found the following directories:"
echo "$matches"
echo

if [ "$MODE" != "-f" ]; then
    echo "Preview mode only."
    echo "Run with '-f' to delete:"
    echo "  cleanup_generated_dirs.sh . -f"
    exit 0
fi

echo "Deleting..."
find "$TARGET_DIR" -type d \( \
    -name ".ipynb_checkpoints" -o \
    -name "isochrones" -o \
    -name "__pycache__" \
\) -exec rm -rf {} +

echo "Done."