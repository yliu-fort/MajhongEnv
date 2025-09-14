#!/usr/bin/env bash
set -euo pipefail

# ===  ===
PY_VER="3.10"
ENV_NAME="riichi_data_builder"
REPO_URL="https://github.com/yliu-fort/MajhongEnv.git"
REPO_DIR="/data/MajhongEnv"

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

echo
echo "[] Init done."
echo "To start working:"
echo "cd $REPO_DIR"

cd $REPO_DIR
nohup python dataset_builder_zarr.py > log.out 2>&1 &