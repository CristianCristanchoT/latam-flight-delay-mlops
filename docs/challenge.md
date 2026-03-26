
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



### 1.2 `preprocess()` Implementation (`challenge/model.py`)

#### What was implemented

The `preprocess()` method of `DelayModel` was implemented by translating the feature engineering logic from the notebook into production-ready helper functions at module level:

| Helper function | Origin | Description |
|---|---|---|
| `_get_period_day(date)` | Notebook Section 4 | Classifies departure time as `mañana`, `tarde`, or `noche` |
| `_is_high_season(fecha)` | Notebook Section 4 | Returns `1` if the date falls within a peak travel season, `0` otherwise |
| `_get_min_diff(row)` | Notebook Section 4 | Computes the difference in minutes between scheduled (`Fecha-I`) and actual (`Fecha-O`) departure |

The method performs the following steps in order:

1. **Generate derived columns** — `period_day`, `high_season`, `min_diff`, and `delay` (binary: `1` if `min_diff > 15`, else `0`)
2. **One-hot encode** — `OPERA`, `TIPOVUELO`, and `MES` using `pd.get_dummies()`
3. **Filter to top 10 features** — defined in `TOP_FEATURES`; any missing column is added with value `0`
4. **Return** — `(features, target)` if `target_column` is provided; otherwise `features` only

#### Constants extracted

Two module-level constants were introduced to avoid magic numbers and repeated literals:

```python
THRESHOLD_IN_MINUTES = 15  # delay threshold from notebook Section 4

TOP_FEATURES = [            # top 10 features selected in notebook Section 6
    "OPERA_Latin American Wings", "MES_7", "MES_10", "OPERA_Grupo LATAM",
    "MES_12", "TIPOVUELO_I", "MES_4", "MES_11", "OPERA_Sky Airline", "OPERA_Copa Air",
]
```

#### Bug fixed: wrong type hint in method signature

The original skeleton used `Union(...)` (function call syntax) instead of `Union[...]` (subscript syntax), which is a Python type-hint error:

```python
# Original (INCORRECT — Union is not callable)
) -> Union(Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame):

# Fixed
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
```

#### Bug fixed: wrong data path in test `setUp`

The test file referenced the CSV with a path relative to the project root (`../data/data.csv`), but pytest resolves paths relative to the working directory where it is invoked. The path was corrected to be relative to `tests/model/`:

```python
# Original (INCORRECT)
self.data = pd.read_csv(filepath_or_buffer="../data/data.csv")

# Fixed
self.data = pd.read_csv(filepath_or_buffer="../../data/data.csv")
```

#### Tests passing

```
test_model_preprocess_for_training  PASSED
test_model_preprocess_for_serving   PASSED
```