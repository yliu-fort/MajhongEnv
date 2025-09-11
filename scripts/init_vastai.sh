#!/usr/bin/env bash
set -euo pipefail

# ===  ===
PY_VER="3.10"
ENV_NAME="visual"
REPO_URL="https://github.com/yliu-fort/MajhongEnv.git"
REPO_DIR="/workspace/MajhongEnv"

echo "[*] Updating packages..."
sudo apt-get update -y
pip install --upgrade pip

echo "[*] Cloning repo..."
# Remove old repo and clone fresh
rm -rf "$REPO_DIR"
git clone "$REPO_URL" "$REPO_DIR"
cd "$REPO_DIR" || { echo " : $REPO_DIR"; exit 1; }

echo "[*] Installing project requirements..."
if [[ -f requirements.txt ]]; then
pip install -r requirements.txt
fi

echo "[*] Running unittests..."
python -m unittest discover -s ./tests -v || true