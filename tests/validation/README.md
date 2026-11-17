# Validation Suite

Cross-validation protocol against **lavaan** (proposal Section 3.1 & 4.4):
every estimator and fit index is validated against reference implementations
on benchmark datasets before release.

## Layout
- `test_validation.py` — Python-side tests (parameter recovery + comparison).
- `fit_lavaan.R` — reference fits the same synthetic datasets in lavaan.

## Running
```bash
# Full suite (requires R + lavaan installed; otherwise auto-skipped)
pytest tests/validation -v

# Just the R-independent parameter recovery check
pytest tests/validation::test_parameter_recovery -v
```
