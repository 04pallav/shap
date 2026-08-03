# Open PRs — why these fixes matter

Author: 04pallav · Fork: [github.com/04pallav/shap](https://github.com/04pallav/shap) · Upstream: [shap/shap](https://github.com/shap/shap)

These three PRs share one theme: **SHAP is used in production ML pipelines, but common real-world inputs cause hard crashes instead of useful explanations or clear errors.** Each fix removes a footgun that blocks or silently breaks explainability for models people actually deploy.

---

## PR #5086 — Clear `TypeError` for list input in TreeExplainer

**Branch:** `fix/issue-4473` · **Issue:** [#4473](https://github.com/shap/shap/issues/4473) · **PR:** [#5086](https://github.com/shap/shap/pull/5086)

### Do we even need a fix?

**Yes — same file, same function, but a different input type than #5073.**

On current `master`, `_validate_inputs` already handles pandas (`DataFrame`/`Series` → `to_numpy`) at line ~466. NumPy arrays pass through. But plain Python lists are neither — they skip the pandas branch and hit:

```
TreeExplainer._validate_inputs
  → (list skips pandas to_numpy branch)
  → len(X.shape)                    # AttributeError: 'list' has no attribute 'shape'
```

Common in notebooks: `explainer.shap_values(list(X[0]))` or `X.values.tolist()`.

**When it triggers:** any `shap_values()` / `__call__()` with a plain `list` as `X`. Background data and model type don't matter.

**Workaround without this PR:** wrap yourself — `np.array(sample)` or `np.asarray(sample)`. Easy, but the error today is misleading (`AttributeError` deep in validation, not "wrong type").

**Design choice in this PR:** raise a clear `TypeError`, don't auto-convert lists. Lists are ambiguous (1D vs 2D). SHAP already accepts numpy and pandas; explicit conversion keeps behavior predictable — same philosophy as rejecting unsupported types rather than guessing.

### Code paths changed

| Location | Before | After | Why |
|---|---|---|---|
| `_tree.py` `_validate_inputs` (~467) | pandas `to_numpy`, then immediately `len(X.shape)` | add `if not isinstance(X, np.ndarray): raise TypeError(...)` after pandas branch | Fail fast with actionable message before `.shape` access |

Two lines. Placed right after the existing nullable-dtype conversion — same function, same pattern as #5073's instance-`X` fix.

### Tests added

`test_list_input_raises_type_error` in `tests/explainers/test_tree.py`:
- single row as list: `shap_values(list(X.iloc[0]))` → `TypeError` matching `"must be a numpy array"`
- 2D as nested list: `shap_values(X.values.tolist())` → same `TypeError`

### Line-by-line walkthrough

#### The problem

`shap_values(X)` enters `_validate_inputs`. On master:

```465:471:shap/shap/explainers/_tree.py
        if isinstance(X, (pd.Series, pd.DataFrame)):
            X = X.to_numpy(dtype=self.model.input_dtype, na_value=np.nan)
        flat_output = False
        if len(X.shape) == 1:    # ← crashes here if X is a list
```

- **numpy** → has `.shape` ✓
- **pandas** → converted, has `.shape` ✓
- **list** → skips pandas branch → `AttributeError: 'list' object has no attribute 'shape'`

#### The fix — `_tree.py` (2 lines added)

Inserted after pandas conversion, before `len(X.shape)`:

| Line | Code | What it does |
|---|---|---|
| 1 | `if not isinstance(X, np.ndarray):` | After pandas branch, is `X` still not a numpy array? |
| 2 | `raise TypeError(f"X must be a numpy array (or pandas DataFrame/Series), not {type(X).__name__}")` | Clear error (e.g. `not list`) instead of `AttributeError` |

#### The fix — `test_tree.py`

| Line | Code | What it does |
|---|---|---|
| 1 | `def test_list_input_raises_type_error():` | New test |
| 2 | docstring | Documents intent: `TypeError`, not `AttributeError` |
| 3–6 | load data, fit `DecisionTreeRegressor`, create explainer | Minimal setup |
| 7–8 | `pytest.raises(TypeError, match="must be a numpy array")` | Assert error type and message |
| 9 | `shap_values(list(X.iloc[0]))` | Case 1: one row as 1D list |
| 10–11 | second `pytest.raises` block | Same assertion |
| 12 | `shap_values(X.values.tolist())` | Case 2: full 2D nested list |

**Summary:** Problem = list reaches `.shape` unchecked. Fix = type guard after pandas conversion. Test = 1D and 2D list inputs.

---

---

## PR #5073 — Nullable pandas dtypes in TreeExplainer background data

**Branch:** `fix/issue-4911-nullable-dtypes` · **Issue:** [#4911](https://github.com/shap/shap/issues/4911) · **PR:** [#5073](https://github.com/shap/shap/pull/5073)

### Do we even need a fix?

**Yes — but only on one code path that master already half-fixed elsewhere.**

On current `master`, nullable dtypes on **instance data** (`X` passed to `shap_values`) already work. `_validate_inputs` converts DataFrames with `to_numpy(dtype=self.model.input_dtype, na_value=np.nan)` (line ~467), and `test_nullable_pandas_dtype` covers that (#4011).

The gap is **background data** (`data=` in `TreeExplainer.__init__`). That path still does `self.data = data.values` (line ~254). For nullable columns (`Int64`, `Float64`, etc.), `.values` returns **object dtype**, not float. The crash happens at **init time**, before any `shap_values` call:

```
TreeExplainer.__init__
  → self.data = data.values          # object array
  → TreeEnsemble(model, self.data, ...)
    → _cext.dense_tree_update_weights(...)
      → TypeError: Cannot cast array data from dtype('O') to dtype('float64')
```

**When it triggers:** only when you pass a nullable-dtype DataFrame as `data=` (interventional / `feature_perturbation="auto"` with background). `tree_path_dependent` with no background is unaffected.

**Workaround without this PR:** cast before passing — `data=X.to_numpy(dtype=float, na_value=np.nan)` or `X.fillna(0)`. So it's not unblockable, but it's inconsistent: instance `X` is handled inside SHAP, background `data` is not.

### Code paths changed

| Location | Before | After | Why |
|---|---|---|---|
| `TreeExplainer.__init__` (~239) | DataFrame passed through to masker as-is | `data.to_numpy(dtype=float, na_value=np.nan)` immediately after reading column names | Converts background before masker/`self.data` assignment; `pd.NA` → `np.nan` |
| `TreeExplainer.__init__` (~253–254) | `if isinstance(data, pd.DataFrame): self.data = data.values` | Removed (dead code) | Data is always numpy after early conversion; `.values` branch unreachable |
| `_build_explanation` (~424) | `X_data = X.values` | `X_data = X.to_numpy(dtype=float, na_value=np.nan)` | Populates `Explanation.data` metadata; same object-dtype trap, but only affects output object, not the C compute path |

The init-time fix is the critical one. `_build_explanation` is consistency — without it, explanations compute fine but `Explanation.data` could still be an object array.

### Test added

`test_nullable_pandas_dtype_background_data` — fits a sklearn tree on nullable `Float64`/`Int64`, passes nullable background to `TreeExplainer(model, data=X_bg)`, asserts `shap_values` returns shape `(5, 2)` without C-extension `TypeError`.

### Line-by-line walkthrough

#### The problem

Master already handles nullable dtypes for **instance data** (`X` in `shap_values`):

```465:467:shap/shap/explainers/_tree.py
        if isinstance(X, (pd.Series, pd.DataFrame)):
            X = X.to_numpy(dtype=self.model.input_dtype, na_value=np.nan)
```

But **background data** (`data=` in `TreeExplainer.__init__`) still uses `.values`:

```253:254:shap/shap/explainers/_tree.py
        if isinstance(data, pd.DataFrame):
            self.data = data.values
```

For nullable pandas columns (`Int64`, `Float64`), `.values` produces an **object** array. That flows into:

```
TreeExplainer.__init__
  → self.data = data.values              # object dtype
  → TreeEnsemble(model, self.data, ...)
    → _cext.dense_tree_update_weights(...)
      → TypeError: Cannot cast array data from dtype('O') to dtype('float64')
```

Crash is at **init** — before `shap_values` is called. `tree_path_dependent` with no background is unaffected.

**Known gap in current PR diff:** conversion is inside `elif isinstance(data, pd.DataFrame)` paired with `feature_names` — skipped when `feature_names=` is passed explicitly. Separate `if` would close that.

---

#### The fix — `_tree.py` `__init__` (1 line added, 2 lines removed)

**Change 1 — early conversion (1 line added after column names)**

| Line | Code | What it does |
|---|---|---|
| 1 | `data = data.to_numpy(dtype=float, na_value=np.nan)` | Convert nullable DataFrame → float numpy **before** masker/`self.data`; `pd.NA` → `np.nan` |

Inserted in the `elif isinstance(data, pd.DataFrame)` block (after `self.data_feature_names = list(data.columns)`). `masker = data` on the next line now receives numpy, not a DataFrame with object `.values`.

**Change 2 — remove dead `.values` branch (2 lines removed)**

| Before | After | What it does |
|---|---|---|
| `if isinstance(data, pd.DataFrame): self.data = data.values` | removed | Unreachable after early conversion — this was the crash site |
| `elif isinstance(data, DenseData):` | `if isinstance(data, DenseData):` | Promoted to first branch since DataFrame case is gone |

`self.data` is now always numpy (or `DenseData.data`) when it came from a nullable DataFrame.

---

#### The fix — `_tree.py` `_build_explanation` (1 line changed)

Runs **after** `shap_values` computes — populates `Explanation.data` for plots/metadata only.

| Before | After | What it does |
|---|---|---|
| `X_data = X.values` | `X_data = X.to_numpy(dtype=float, na_value=np.nan)` | Same object-dtype trap as background; `Explanation.data` stays float array |

Does **not** fix the #4911 init crash — that's the `__init__` change above. This is consistency with the instance-`X` path in `_validate_inputs`.

---

#### The fix — `test_tree.py` (new test, line by line)

| Lines | Code | What it does |
|---|---|---|
| 1 | `def test_nullable_pandas_dtype_background_data():` | New test — background path, not instance path |
| 2–6 | docstring | Documents #4911: `.values` → object → C extension `TypeError` |
| 7–14 | build `X` with `Float64`/`Int64` columns | Training data with nullable dtypes |
| 15–18 | fit `DecisionTreeClassifier` | Simple tree model |
| 20–27 | build `X_bg` (3-row background) | Background with nullable dtypes |
| 29–31 | dtype assertions | Precondition: columns are actually nullable |
| 33–35 | `TreeExplainer(model, data=X_bg)` then `shap_values(X.iloc[:5])` | Init + explain — would crash at init on master |
| 36 | `assert sv.shape == (5, 2)` | 5 rows, 2 features — success |

---

**Summary:** Problem = background `data` still uses `.values` while instance `X` already uses `to_numpy`. Fix = convert early in `__init__`, remove dead `.values` branch, align `_build_explanation`. Test = nullable background through full init + `shap_values`.

---

---

## PR #5078 — `shap.Explainer` hard-fails on InterpretML EBMs with interaction terms

**Branch:** `fix/issue-4942-ebm-interactions` · **Issue:** [#4942](https://github.com/shap/shap/issues/4942) · **PR:** [#5078](https://github.com/shap/shap/pull/5078)

### Do we even need a fix?

**Yes — and it takes two surgical changes, because there are two failures in the chain.**

Default InterpretML EBMs have `interactions != 0`. `shap.Explainer(ebm, X)` is supposed to auto-pick an explainer, then fall back to model-agnostic if no specialized path fits.

**Failure 1 — auto-selection crashes instead of skipping:**

```
Explainer.__init__  (algorithm="auto")
  → AdditiveExplainer.supports_model_with_masker(ebm, masker)
    → if model.interactions != 0:
         raise NotImplementedError(...)   # master
```

`supports_model_with_masker` is a *selection gate* — it should return `True`/`False`, not raise. Raising aborts the whole `Explainer` call before any fallback runs.

**Failure 2 — even after returning `False`, fallback still breaks:**

```
Explainer.__init__  (after supports returns False for interactions EBM)
  → linear? no  → tree? no  → additive? no
  → elif callable(self.model):   # EBM object is NOT callable
  → else: raise TypeError("model is not callable...")
```

So `return False` alone is insufficient — same pattern as transformers: wrap the model into a callable and re-invoke `__init__`.

**Third gap on master:** `ExplainableBoostingRegressor` is never handled in `AdditiveExplainer.__init__` (only `ExplainableBoostingClassifier` is). EBR with `interactions=0` can't use the fast additive path at all.

**When it triggers:** any `shap.Explainer(ebm, X)` where EBM has default interaction terms, or any EBR model.

**Workaround without this PR:** bypass `Explainer` — use `shap.PermutationExplainer(ebm.predict, masker)` manually. Doable, but defeats the unified API.

**What this PR does *not* do:** implement interaction effects in `AdditiveExplainer`. That would be wrong (additive assumes first-order only). Fallback to permutation/exact is the correct behavior.

### Code paths changed

| Location | Before | After | Why |
|---|---|---|---|
| `_additive.py` `supports_model_with_masker` (~116) | `raise NotImplementedError` when `interactions != 0` | `return False` | Lets auto-selector skip additive path gracefully |
| `_additive.py` `supports_model_with_masker` (new) | EBR not checked | `return False` if interactions, else `return True` | EBR gets same gate as EBC |
| `_additive.py` `__init__` (new) | no EBR branch | `elif EBR: self.model = model.predict` | Fast additive path for EBR with `interactions=0` |
| `_explainer.py` fallback `else` (~216) | `raise TypeError` for non-callable model | detect EBC/EBR, wrap as `decision_function`/`predict`, `return self.__init__(...)` | Same retry pattern as transformers pipeline wrapping (~136–158) |

Both files are required — fixing only `_additive.py` still leaves interaction EBMs hitting `TypeError` in the fallback.

### Tests added

New file `tests/explainers/test_additive.py`:
- `test_supports_classifier_no_interaction` — EBC `interactions=0` → `True`
- `test_supports_classifier_with_interaction` — EBC default interactions → `False`
- `test_supports_regressor_no_interaction` — EBR `interactions=0` → `True`
- `test_additive_explainer_with_ebm_regressor` — end-to-end additive path for EBR

### Line-by-line walkthrough

#### The problem

User calls `shap.Explainer(ebm, X)` with a default InterpretML EBM (`interactions != 0`). `Explainer.__init__` runs auto-selection (`algorithm="auto"`):

```185:194:shap/shap/explainers/_explainer.py
            if algorithm == "auto":
                if explainers.LinearExplainer.supports_model_with_masker(model, self.masker):
                    algorithm = "linear"
                elif explainers.TreeExplainer.supports_model_with_masker(model, self.masker):
                    algorithm = "tree"
                elif explainers.AdditiveExplainer.supports_model_with_masker(model, self.masker):
                    algorithm = "additive"
```

On master, step 3 for an EBM with interactions:

```115:118:shap/shap/explainers/_additive.py
        if safe_isinstance(model, "interpret.glassbox.ExplainableBoostingClassifier"):
            if model.interactions != 0:
                raise NotImplementedError("Need to add support for interaction effects!")
            return True
```

**Crash 1:** `NotImplementedError` — selection gate raises instead of returning `False`, so fallback never runs.

Even after fixing that, auto-selection continues:

```196:220:shap/shap/explainers/_explainer.py
                elif callable(self.model):
                    ...
                else:
                    raise TypeError(
                        "The passed model is not callable and cannot be analyzed directly with the given masker! Model: "
                        + str(model)
                    )
```

**Crash 2:** EBM objects are not callable (`callable(ebm)` is `False`), so you hit `TypeError` in the `else` branch.

**Gap 3:** `ExplainableBoostingRegressor` is never wired in `AdditiveExplainer.__init__` (only `ExplainableBoostingClassifier` sets `self.model = model.decision_function`).

---

#### The fix — `_additive.py`

**Change 1 — `__init__`: wire up EBR (3 lines added after EBC block)**

| Line | Code | What it does |
|---|---|---|
| 1 | `elif safe_isinstance(model, "interpret.glassbox.ExplainableBoostingRegressor"):` | Same pattern as EBC above — detect InterpretML regressor |
| 2 | `self.model = model.predict` | `AdditiveExplainer` needs a **callable**; EBR exposes `predict`, not `decision_function` |

Without this, EBR with `interactions=0` can't use the fast additive path even when `supports_model_with_masker` returns `True`.

**Change 2 — `supports_model_with_masker`: stop raising, gate both EBM types**

| Line | Before | After | What it does |
|---|---|---|---|
| EBC + interactions | `raise NotImplementedError(...)` | `return False` | "Additive can't handle this" — skip, don't crash |
| EBC + no interactions | `return True` (unchanged) | `return True` | Use fast `AdditiveExplainer` |
| EBR block (new) | not present | `if interactions != 0: return False` else `return True` | Same gate for regressor |

`return False` means auto-selection tries the next path (model-agnostic fallback).

---

#### The fix — `_explainer.py` (fallback `else` block)

Replaces a single `raise TypeError` with detect-and-retry — same idea as transformers pipeline wrapping (~136–158).

| Line | Code | What it does |
|---|---|---|
| 1 | `# Check if model is an interpret EBM — wrap into a callable and retry` | Comment: why we're here |
| 2 | `if safe_isinstance(self.model, "interpret.glassbox.ExplainableBoostingClassifier"):` | EBM classifier fell through because not callable |
| 3–12 | `return self.__init__(self.model.decision_function, self.masker, ...)` | Restart `__init__` with **callable** `decision_function` as `model`; auto-selection can now pick permutation/exact |
| 13 | `elif safe_isinstance(self.model, "interpret.glassbox.ExplainableBoostingRegressor"):` | Same for regressor |
| 14–23 | `return self.__init__(self.model.predict, self.masker, ...)` | Restart with `predict` as callable |
| 24–28 | `else: raise TypeError(...)` | Original error — only for models we still can't wrap |

**Why re-call `__init__`?** Second pass sees a callable model → `elif callable(self.model)` succeeds → picks `permutation` or `exact` based on feature count.

---

#### The fix — `test_additive.py` (new file, line by line)

| Lines | Code | What it does |
|---|---|---|
| 1–4 | imports | `numpy`, `pytest`, `shap` |
| 7–16 | `ebm_classifier_no_interaction` fixture | Train EBC with `interactions=0`; skip test if `interpret` not installed |
| 19–28 | `ebm_classifier_with_interaction` fixture | Train EBC with `interactions=1` (default-like) |
| 31–40 | `ebm_regressor_no_interaction` fixture | Train EBR with `interactions=0` |
| 43–44 | `test_supports_classifier_no_interaction` | Assert `supports_model_with_masker` → `True` for EBC no interactions |
| 47–48 | `test_supports_classifier_with_interaction` | Assert → `False` for EBC with interactions |
| 51–52 | `test_supports_regressor_no_interaction` | Assert → `True` for EBR no interactions |
| 55–59 | `test_additive_explainer_with_ebm_regressor` | End-to-end: `AdditiveExplainer(ebr, masker)` → `shap_values` shape matches `X` |

---

**Summary:** Problem = two crashes in the auto-selection chain (`NotImplementedError` then `TypeError`) plus missing EBR wiring. Fix = `return False` to skip additive, wrap EBM into callable and retry `__init__` for fallback, add EBR to `__init__`. Test = gate logic + one EBR end-to-end path.

---

## Summary

| PR | Size | Production relevance | One-line pitch |
|---|---|---|---|
| #5086 | Small (~16 lines) | Medium — API robustness, faster debugging | Libraries at SHAP's scale should give clear errors, not opaque crashes |
| #5073 | Medium (~40 lines) | High — blocks tree explainability on modern pandas data | Nullable dtypes are the default in real ETL; SHAP should accept them |
| #5078 | Big (~98 lines) | High — blocks unified API on a major model class | EBMs are deployed for interpretability; `Explainer` must not crash on them |

All three are **bug fixes on the critical path to explainability**, not cosmetic changes.

---

## Next notes to add

- [ ] Maintainer feedback / review status per PR
- [ ] Test commands to reproduce each issue locally
- [ ] Before/after code snippets
- [ ] Merge blockers and follow-up PR ideas
