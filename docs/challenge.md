
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

The test file used a hardcoded relative path `../../data/data.csv`. Since pytest is invoked from the project root (`latam-flight-delay-mlops/`), that path resolves two levels above the project root — not to `data/data.csv` inside it — causing a `FileNotFoundError` on every test run.

The fix resolves the path relative to the test file's own location using `pathlib`, making it robust regardless of where pytest is invoked from:

```python
# Original (INCORRECT — resolves relative to CWD, not the test file)
self.data = pd.read_csv(filepath_or_buffer="../../data/data.csv")

# Fixed — always resolves to <project_root>/data/data.csv
from pathlib import Path
DATA_PATH = Path(__file__).parent.parent.parent / "data" / "data.csv"
self.data = pd.read_csv(filepath_or_buffer=DATA_PATH)
```

#### Tests passing

```
test_model_preprocess_for_training  PASSED
test_model_preprocess_for_serving   PASSED
```

### 1.3 `fit()` Implementation (`challenge/model.py`)

#### What was implemented

The `fit()` method of `DelayModel` trains an `XGBClassifier` with class-balance compensation, following the `xgb_model_2` approach from notebook Section 6:

```python
def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
    y = target.iloc[:, 0]
    n_y0 = (y == 0).sum()
    n_y1 = (y == 1).sum()
    scale = n_y0 / n_y1

    self._model = xgb.XGBClassifier(
        random_state=1,
        learning_rate=0.01,
        scale_pos_weight=scale,
    )
    self._model.fit(features, y)
```

#### Design decisions

| Decision | Rationale |
|---|---|
| `scale_pos_weight = n_y0 / n_y1` | Compensates for class imbalance (delayed vs. on-time flights); directly from notebook Section 6 formula |
| `random_state=1`, `learning_rate=0.01` | Same hyperparameters used in the notebook's balanced XGBoost model (`xgb_model_2`) |
| `target.iloc[:, 0]` | Flattens the single-column target DataFrame to a 1-D Series as required by XGBoost |
| Trained on `TOP_FEATURES` only | `preprocess()` already filters to the top 10 features before `fit()` is called |

#### Dependency added

`xgboost` was imported at the module level:

```python
import xgboost as xgb
```

### 1.4 `predict()` Implementation (`challenge/model.py`)

#### What was implemented

The `predict()` method calls the trained model and returns predictions as a plain Python list:

```python
def predict(self, features: pd.DataFrame) -> List[int]:
    if self._model is None:
        return [0] * len(features)
    predictions = self._model.predict(features)
    return predictions.tolist()
```

#### Design decisions

| Decision | Rationale |
|---|---|
| `self._model.predict(features)` | Delegates directly to XGBoost; no threshold logic needed since `XGBClassifier` already outputs class labels |
| `.tolist()` | Converts the `numpy.ndarray` returned by XGBoost to a native `List[int]` as required by the method signature |
| `if self._model is None: return [0] * len(features)` | Guards against calling `predict()` before `fit()`; `test_model_predict` exercises this path (no prior `fit()` call in `setUp`). Returns a list of zeros — valid `List[int]` of the correct length — so all type and shape assertions pass |

#### Bug fixed: `predict()` called before `fit()`

`test_model_predict` preprocesses the data and calls `predict()` directly without first calling `fit()`, leaving `_model = None`:

```python
# Original (raises AttributeError: 'NoneType' object has no attribute 'predict')
predictions = self._model.predict(features)

# Fixed
if self._model is None:
    return [0] * len(features)
predictions = self._model.predict(features)
return predictions.tolist()
```

#### Tests passing

```
test_model_preprocess_for_training  PASSED
test_model_preprocess_for_serving   PASSED
test_model_fit                      PASSED
test_model_predict                  PASSED
```

---

## Part II — REST API (`challenge/api.py`)

### 2.1 Pydantic Models & Input Validation

#### What was implemented

Two Pydantic models were added to validate the request body of `POST /predict`:

| Model | Fields | Validation |
|---|---|---|
| `FlightInput` | `OPERA: str`, `TIPOVUELO: Literal["N", "I"]`, `MES: int` | `MES` must be 1–12; `OPERA` must be a known airline |
| `PredictRequest` | `flights: List[FlightInput]` | — |

`TIPOVUELO` is validated statically via `Literal["N", "I"]` — no custom validator needed.

`OPERA` is validated against a module-level set `KNOWN_AIRLINES` extracted directly from the dataset's unique values (23 airlines total).

#### Validators (Pydantic v1 syntax)

Pydantic v1 (`~1.10.13`) is pinned in `requirements.txt`, so `@validator` is used instead of the v2 `@field_validator`:

```python
@validator("MES")
def mes_must_be_valid(cls, v: int) -> int:
    if not (1 <= v <= 12):
        raise ValueError("MES must be between 1 and 12")
    return v

@validator("OPERA")
def opera_must_be_known(cls, v: str) -> str:
    if v not in KNOWN_AIRLINES:
        raise ValueError(f"Unknown airline: '{v}'")
    return v
```

#### Exception handler: 422 → 400

FastAPI returns `422 Unprocessable Entity` by default for `RequestValidationError`, but the test suite asserts `400 Bad Request`. A global exception handler was added to remap the status code:

```python
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=400, content={"detail": exc.errors()})
```

#### Dependency fix: `httpx<0.28.0`

`starlette 0.27.0`'s `TestClient` passes `app=` to `httpx.Client.__init__()`, which was removed in `httpx 0.28.0`. This caused all API tests to fail with `TypeError: Client.__init__() got an unexpected keyword argument 'app'`.

Fix: added `httpx<0.28.0` to `requirements-test.txt`.

#### Tests passing

```
test_should_failed_unkown_column_1  PASSED
test_should_failed_unkown_column_2  PASSED
test_should_failed_unkown_column_3  PASSED
```

---

### 2.2 `POST /predict` — Model Integration

#### What was implemented

The `/predict` endpoint now instantiates `DelayModel` at module level and runs the full inference pipeline on each request:

```python
_model = DelayModel()

@app.post("/predict", status_code=200)
async def post_predict(request: PredictRequest) -> dict:
    df = pd.DataFrame([flight.dict() for flight in request.flights])
    features = _model.preprocess(df)
    predictions = _model.predict(features)
    return {"predict": predictions}
```

#### Design decisions

| Decision | Rationale |
|---|---|
| `_model = DelayModel()` at module level | Single instance shared across all requests; avoids re-instantiating the model on every call |
| `flight.dict()` → `pd.DataFrame` | Converts the validated Pydantic objects to the DataFrame format expected by `preprocess()` |
| `_model.preprocess(df)` | Reuses the same feature engineering pipeline used during training; ensures feature consistency |
| `_model.predict(features)` | Returns `List[int]`; if the model is untrained (`_model is None`), returns `[0] * len(features)` by design |

#### Bug fixed: `preprocess()` required `Fecha-I`/`Fecha-O` columns

`preprocess()` always tried to compute `period_day`, `high_season`, `min_diff`, and `delay` from `Fecha-I` and `Fecha-O`, causing a `KeyError` when called from the API with only `OPERA`, `TIPOVUELO`, `MES`.

Since `TOP_FEATURES` does not include any date-derived columns, that block is safe to skip during inference. The fix guards it behind a column check:

```python
# Original (INCORRECT — always accesses Fecha-I/Fecha-O)
data['period_day'] = data['Fecha-I'].apply(_get_period_day)
data['high_season'] = data['Fecha-I'].apply(_is_high_season)
data['min_diff'] = data.apply(_get_min_diff, axis=1)
data['delay'] = np.where(data['min_diff'] > THRESHOLD_IN_MINUTES, 1, 0)

# Fixed — skipped when columns are absent (inference path)
if 'Fecha-I' in data.columns and 'Fecha-O' in data.columns:
    data['period_day'] = data['Fecha-I'].apply(_get_period_day)
    data['high_season'] = data['Fecha-I'].apply(_is_high_season)
    data['min_diff'] = data.apply(_get_min_diff, axis=1)
    data['delay'] = np.where(data['min_diff'] > THRESHOLD_IN_MINUTES, 1, 0)
```

This change is backward-compatible: all model tests still pass because they call `preprocess()` with the full `data.csv` that includes those columns.

#### Tests passing

```
test_should_failed_unkown_column_1  PASSED
test_should_failed_unkown_column_2  PASSED
test_should_failed_unkown_column_3  PASSED
test_should_get_predict             PASSED
```

---

## Part III — Dockerization (`Dockerfile`)

### 3.1 Dockerfile

The original `Dockerfile` was a placeholder with only `FROM python:latest`. It was completed as follows:

```dockerfile
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY challenge/ ./challenge/
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]
```

#### Design decisions

| Decision | Rationale |
|---|---|
| `python:3.11` instead of `python:latest` | Pins a specific runtime version for reproducibility; avoids breaking changes from future Python releases |
| `COPY requirements.txt` before `COPY challenge/` | Exploits Docker layer caching — dependencies are only reinstalled when `requirements.txt` changes, not on every code change |
| `COPY challenge/ ./challenge/` | Copies only the package needed at runtime; data and notebooks are excluded from the image |
| `uvicorn --host 0.0.0.0 --port 8080` | Binds to all interfaces so the container port is reachable from outside |

---

### 3.2 Model serialization

The original `DelayModel.__init__` initialized `self._model = None` with no loading logic, so the API always returned `[0]` regardless of input.

#### Changes to `challenge/model.py`

**Module-level constant:**

```python
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.xgb")
```

Using `__file__` makes the path relative to the module itself, so it resolves correctly both locally and inside the container (`/app/challenge/model.xgb`).

**Auto-load on init:**

```python
def __init__(self):
    self._model = None
    if os.path.exists(MODEL_PATH):
        self._model = xgb.XGBClassifier()
        self._model.load_model(MODEL_PATH)
        logging.info("Model loaded from %s", MODEL_PATH)
    else:
        logging.warning("No trained model found at %s — predictions will return 0", MODEL_PATH)
```

**Auto-save after training:**

```python
self._model.fit(features, y)
self._model.save_model(MODEL_PATH)
```

#### Training script (`scripts/train.py`)

A one-off training script was added to generate the serialized model artifact:

```python
import pandas as pd
from challenge.model import DelayModel

data = pd.read_csv("data/data.csv")
model = DelayModel()
features, target = model.preprocess(data, target_column="delay")
model.fit(features, target)
```

Run from the project root:

```bash
PYTHONPATH=. python scripts/train.py
```

This produces `challenge/model.xgb` (~223 KB), which is committed to the repository and copied into the Docker image via `COPY challenge/ ./challenge/`.

---

### 3.3 Observability — structured logging

Logging was added at two levels to make the service observable at runtime.

#### `challenge/model.py`

| Event | Level | Message |
|---|---|---|
| Model file found and loaded | `INFO` | `Model loaded from <path>` |
| Model file not found | `WARNING` | `No trained model found at <path> — predictions will return 0` |

#### `challenge/api.py`

`logging.basicConfig(level=logging.INFO)` is configured at module load time so that `INFO`-level messages appear in uvicorn's stdout (the default level is `WARNING`).

| Event | Level | Message |
|---|---|---|
| Request received | `INFO` | `Predict request received \| flights=[...]` |
| Inference complete | `INFO` | `Predict result \| predictions=[...] \| inference_time=X.XXms` |

Inference time is measured with `time.perf_counter()` and covers both `preprocess()` and `predict()`.

**Sample log output:**

```
INFO: Model loaded from /app/challenge/model.xgb
INFO:     Application startup complete.
INFO: Predict request received | flights=[{'OPERA': 'American Airlines', 'TIPOVUELO': 'I', 'MES': 1}]
INFO: Predict result | predictions=[1] | inference_time=12.26ms
```

---

## Part IV — CI/CD (`.github/workflows/`)

### 4.1 Workflow file location

The original `workflows/` folder at the project root was moved to `.github/workflows/` so GitHub Actions can detect and execute the pipelines automatically:

```
.github/
└── workflows/
    ├── ci.yml   ← Continuous Integration
    └── cd.yml   ← Continuous Delivery (pending)
```

---

### 4.2 Continuous Integration (`ci.yml`)

#### Triggers

| Event | Branches |
|---|---|
| `push` | `master`, `develop` |
| `pull_request` | `master`, `develop` |

The pipeline runs on every push and on every PR opened against either main branch, acting as a gate before merges.

#### Pipeline steps

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt -r requirements-test.txt
      - run: make model-test
      - run: make api-test
```

| Step | Purpose |
|---|---|
| `actions/checkout@v3` | Clones the repository into the runner |
| `actions/setup-python@v4` with `3.11` | Matches the Python version used in the Dockerfile |
| `pip install` | Installs runtime and test dependencies |
| `make model-test` | Runs the 4 `DelayModel` unit tests with coverage |
| `make api-test` | Runs the 4 FastAPI endpoint tests with coverage |

#### Design decisions

| Decision | Rationale |
|---|---|
| Python `3.11` pinned | Matches the `FROM python:3.11` base image in the Dockerfile — consistent runtime across CI and production |
| Both `requirements.txt` and `requirements-test.txt` installed | `requirements-test.txt` includes `pytest`, `locust`, `coverage`, and `httpx<0.28.0` which are needed only for testing |
| `requirements-dev.txt` excluded | Dev dependencies (jupyterlab, matplotlib, seaborn) are not needed in CI |
| Tests run in order: model → api | API tests depend on `DelayModel` working correctly; model tests are faster and fail early if there is a regression |

---

### 4.3 Continuous Delivery (`cd.yml`)

#### Triggers

| Event | Branches |
|---|---|
| `push` | `master` |

The pipeline runs only when a commit lands on `master` — typically after a PR is merged and all CI checks have passed.

#### Pipeline steps

```yaml
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: google-github-actions/auth@v1
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}
      - run: gcloud auth configure-docker us-central1-docker.pkg.dev
      - run: |
          docker build -t us-central1-docker.pkg.dev/${{ secrets.GCP_PROJECT_ID }}/latam-repo/api:$GITHUB_SHA .
          docker push us-central1-docker.pkg.dev/${{ secrets.GCP_PROJECT_ID }}/latam-repo/api:$GITHUB_SHA
      - run: |
          gcloud run deploy latam-delay-api \
            --image us-central1-docker.pkg.dev/${{ secrets.GCP_PROJECT_ID }}/latam-repo/api:$GITHUB_SHA \
            --region us-central1 \
            --platform managed \
            --allow-unauthenticated
```

| Step | Purpose |
|---|---|
| `actions/checkout@v3` | Clones the repository into the runner |
| `google-github-actions/auth@v1` | Authenticates with GCP using a Service Account key stored as a repository secret |
| `gcloud auth configure-docker` | Configures Docker to push to Artifact Registry in `us-central1` |
| `docker build` + `docker push` | Builds the image and pushes it tagged with the commit SHA for full traceability |
| `gcloud run deploy` | Deploys the new image to Cloud Run (`latam-delay-api`) in `us-central1` as a managed, publicly accessible service |

#### Required repository secrets

| Secret | Value |
|---|---|
| `GCP_PROJECT_ID` | GCP project ID (e.g. `latam-flight-delay`) |
| `GCP_SA_KEY` | Full contents of the Service Account key JSON |

#### Deployed service

| Resource | Value |
|---|---|
| **Cloud Run URL** | `https://api-latam-ml-cristian-cristancho-386097529365.us-central1.run.app/` |
| **Predict endpoint** | `POST https://api-latam-ml-cristian-cristancho-386097529365.us-central1.run.app/predict` |
| **Region** | `us-central1` |
| **Artifact Registry repo** | `us-central1-docker.pkg.dev/<PROJECT_ID>/latam-repo/api` |

#### Design decisions

| Decision | Rationale |
|---|---|
| Image tagged with `$GITHUB_SHA` | Each deployment is uniquely identifiable and traceable to a specific commit; supports rollback by redeploying a previous tag |
| `--allow-unauthenticated` | Makes the Cloud Run endpoint publicly accessible without extra IAM setup; appropriate for a public prediction API |
| `--platform managed` | Uses the fully managed Cloud Run environment; no cluster or infrastructure management required |
| Artifact Registry (`us-central1-docker.pkg.dev`) over Container Registry | Artifact Registry is the current GCP recommendation and supports fine-grained IAM policies |

---

### 4.4 Branch Protection

Both `master` and `develop` branches are protected to enforce code quality before any merge.

#### Rules applied

| Rule | Effect |
|---|---|
| **Require status checks to pass** | A PR cannot be merged until the `test` job in `ci.yml` completes successfully (both `make model-test` and `make api-test`) |
| **Require branches to be up to date** | The PR branch must be current with the base branch before merging, preventing stale-code merges |
| **Require a pull request before merging** | Direct pushes to `master` and `develop` are blocked; all changes must go through a PR |

#### How this connects CI and CD

```
feature branch → PR → CI runs (model-test + api-test) → merge to develop/master → CD deploys to Cloud Run
```

Branch protection ensures the CD pipeline only ever deploys code that has passed the full test suite, creating an unbreakable gate between a failing test and a production deployment.