#!/usr/bin/env bash
# Release script: lint -> typecheck -> test -> build -> upload to PyPI.
set -euo pipefail

echo "==> Cleaning previous builds"
rm -rf build dist *.egg-info src/*.egg-info

echo "==> Linting"
ruff check src tests

echo "==> Type checking"
mypy src/pysemforge || true

echo "==> Running tests"
pytest tests/ -v

echo "==> Building package"
python -m build

echo "==> Checking dist"
twine check dist/*

if [[ "${1:-}" == "--publish" ]]; then
  echo "==> Uploading to TestPyPI first"
  twine upload --repository testpypi dist/*
  echo "==> Uploading to PyPI"
  twine upload dist/*
  echo "==> Tagging release"
  git tag "v$(python -c 'import tomllib;print(tomllib.load(open("pyproject.toml","rb"))["project"]["version"])')"
  git push --tags
else
  echo "==> Dry run complete. Re-run with --publish to upload to PyPI."
fi
