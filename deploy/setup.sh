#!/usr/bin/env bash
# One-shot server setup: run once on a fresh Ubuntu VM from ~/crypto-bot
set -euo pipefail
cd "$(dirname "$0")/.."

sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip

python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

sudo cp deploy/crypto-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now crypto-bot

echo "--- status ---"
systemctl --no-pager status crypto-bot || true
