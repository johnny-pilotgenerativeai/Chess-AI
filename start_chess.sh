#!/bin/bash

# ─────────────────────────────────────────────
#  AI Chess — startup script
#  Edit CHESS_DIR if your path is different
# ─────────────────────────────────────────────
CHESS_DIR="/path/to/Chess-AI-main"
VENV_DIR="$CHESS_DIR/.venv"
SCRIPT="Chess.py"

# Go to project directory
cd "$CHESS_DIR" || { echo "ERROR: Directory $CHESS_DIR not found"; exit 1; }

# Activate virtual environment
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "ERROR: .venv not found at $VENV_DIR"
    exit 1
fi
source "$VENV_DIR/bin/activate"

# Run the Flask app
echo "Starting AI Chess at http://0.0.0.0:5000"
exec python3 "$CHESS_DIR/$SCRIPT"
