#!/usr/bin/env bash
# ========================================
# 🏈 NFL Dashboard Environment Setup (uv)
# ========================================

# Exit on error
set -e

# Project name
PROJECT_NAME="nfl-data-viz"

echo "🚀 Setting up $PROJECT_NAME environment using uv..."

# 1️⃣ Create uv environment
uv venv .venv

# 2️⃣ Activate environment
source .venv/bin/activate

# 3️⃣ Install dependencies
uv pip install -r requirements.txt

# 4️⃣ Optional: verify installation
echo "✅ Installed packages:"
uv pip list

# 5️⃣ Create data folders if missing
mkdir -p data/raw data/processed notebooks src grafana/dashboards

# 6️⃣ Print usage info
echo ""
echo "🎯 Environment ready!"
echo "👉 To activate later, run: source .venv/bin/activate"
echo "👉 To run Jupyter: uv run jupyter lab"
echo "👉 To run scripts:  uv run python src/analysis.py"
echo ""

source venv/bin/activate # (Linux/Mac)