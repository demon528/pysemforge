#!/usr/bin/env bash
# Package pysemforge into a distributable zip + build PyPI artifacts.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/4] Clean caches..."
rm -rf build dist *.egg-info
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type d -name .pytest_cache -exec rm -rf {} +

echo "[2/4] Run tests..."
python -m pytest tests/ -q

echo "[3/4] Build sdist + wheel (hatchling)..."
python -m hatchling build --target sdist --target wheel

echo "[4/4] Pack source zip..."
VERSION=$(python -c "import pysemforge; print(pysemforge.__version__)")
ZIP="pysemforge-${VERSION}-source.zip"
rm -f "../${ZIP}"
zip -rq "../${ZIP}" . -x "*.pyc" "*/__pycache__/*" "*.pytest_cache/*" "build/*" "*.egg-info/*"
echo "Created: ../${ZIP}"
ls -la "../${ZIP}" dist/
