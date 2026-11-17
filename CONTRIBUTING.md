# Contributing to pysemforge

Thanks for your interest! This project follows the workflow described in the
proposal (Section 4). Here's how to contribute:

## Setup

```bash
git clone https://github.com/your-username/pysemforge.git
cd pysemforge
pip install -e .[dev]
```

## Development workflow

1. Create a feature branch: `git checkout -b feat/my-feature`
2. Make your changes, add tests under `tests/`
3. Run the suite: `pytest tests/ -v`
4. Lint: `ruff check src tests`
5. Open a PR — CI will run tests + lavaan cross-validation

## Guidelines

- **Correctness over speed** (proposal §3.1) — every estimator must be
  validated against lavaan on benchmark datasets before merging.
- Keep the public API lavaan-compatible; new features should map to
  familiar R/semTools workflows.
- Update `CHANGELOG.md` and docs with any user-visible change.

## Reporting issues

Use the issue templates under `.github/ISSUE_TEMPLATE/` for bugs and
feature requests.
