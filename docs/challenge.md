
# Challenge Documentation

## Part I — Delay Prediction Model

### 1.1 Notebook Understanding (`exploration.ipynb`)

#### Dependency Updates

The original versions in `requirements.txt` are not compatible with Python 3.11. The following libraries were updated:

| Library | Original Version | Updated Version |
|---------|-----------------|-----------------|
| `fastapi` | 0.86.0 | 0.103.0 |
| `pydantic` | 1.10.2 | 1.10.13 |
| `uvicorn` | 0.15.0 | 0.23.0 |
| `numpy` | 1.22.4 | 1.26.0 |
| `pandas` | 1.3.5 | 2.1.0 |
| `scikit-learn` | 1.3.0 | 1.3.0 |

Additionally, the following dependencies were missing from the original file:

- `xgboost~=2.0.0` — used in the notebook for training but not declared in `requirements.txt`
- `jupyterlab~=4.5.6` — added to `requirements-dev.txt` for notebook execution

---

#### Bug Found: `get_rate_from_column`

**Location:** Section 3 of the notebook — *Data Analysis: Second Sight*

**Description:**

The `get_rate_from_column` function calculates delay rates per category with an inverted division:

```python
# Original code (INCORRECT)
rates[name] = round(total / delays[name], 2)
# Example: 1000 flights / 200 delayed = 5.0  ← this is not a rate

# Applied fix
rates[name] = round(delays[name] / total * 100, 2)
# Example: 200 delayed / 1000 flights * 100 = 20.0%  ← correct delay rate
```

**Consequences:**

- Section 3 charts displayed values > 1 on the Y-axis labeled `Delay Rate [%]`, which is mathematically impossible for a percentage-scale rate
- A taller bar meant *fewer relative delays* (larger values = smaller denominator), so the visual interpretation was the inverse of what was intended
- All absolute rate values reported in that section are incorrect

**Impact on Conclusions:**

The qualitative conclusions in Section 7 of the notebook may still hold if the relative ranking between categories is preserved, but the absolute values are unreliable and the visual interpretation of the charts is incorrect.