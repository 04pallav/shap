# XGBoost categorical — fix notes

**Branch:** `fix/xgboost-categorical-gpu` (fork `04pallav/shap` only — no upstream PR)  
**Back:** [LANE_GPU.md](./LANE_GPU.md) · Tracker: [#4182](https://github.com/shap/shap/issues/4182) categorical item  
**Status:** **XGB #2 done** (CPU `tree_path_dependent` + background). See **Not in this PR** below. XGB #3 / #4–6 still open.

---

## Full case matrix (6 valid cases each)

Grid: **explainer × perturbation × background**. Omitted: `interventional` without background — SHAP emits `FutureWarning` and **silently switches to `tree_path_dependent`** (same as #1 / #4), not real interventional.

### XGBoost categorical (`enable_categorical=True`)

| # | Explainer | Perturbation | Background | Status |
|---|---|---|---|---|
| 1 | CPU | `tree_path_dependent` | No | **Works** — `pred_contribs` shortcut (never hits `_cext`) |
| 2 | CPU | `tree_path_dependent` | Yes | **Done** — `threshold_types=2`, cat codes, additivity vs `output_margin` |
| 3 | CPU | `interventional` | Yes | **Not in this PR** — still `NotImplementedError`; `tree_shap_indep` has no cat routing |
| 4 | GPU | `tree_path_dependent` | No | **Not in this PR** — `_cext_gpu` updated but not validated on CUDA hardware |
| 5 | GPU | `tree_path_dependent` | Yes | **Not in this PR** — no `assert_gpu_matches_cpu` vs CPU #2 |
| 6 | GPU | `interventional` | Yes | **Not in this PR** — unverified |

**Work summary:** one `_tree.py` plumbing change feeds #2–6. Extra CPU C++ for **#3** only (`tree_shap.h`). **#4–6** need real GPU.

### LightGBM categorical (reference — shipped in #4171 / #5020)

**Empirical run (Mar 2026, local CPU; `HouseAgeGroup` cat feature, california subset):**

| # | Explainer | Perturbation | Background | Status |
|---|---|---|---|---|
| 1 | CPU | `tree_path_dependent` | No | **Works** — additivity 0; cat SHAP ≠ 0 |
| 2 | CPU | `tree_path_dependent` | Yes | **Works** — additivity 0; cat SHAP ≠ 0 |
| 3 | CPU | `interventional` | Yes | **Broken** — additivity max err **0.039**; **cat SHAP = 0** |
| 4 | GPU | `tree_path_dependent` | No | **Not run here** — no `_cext_gpu` build on Mac |
| 5 | GPU | `tree_path_dependent` | Yes | **Not run here** |
| 6 | GPU | `interventional` | Yes | **Not run here** |

**Takeaway:** LGBM #3 confirms shared `tree_shap_indep` gap — test LGBM #4–6 on GPU before XGB #4–6.

**Why test LightGBM all 6 first:** plumbing already shipped (#4171/#5020) — establishes what SHAP actually supports vs what we assume. XGB fix should target gaps LGBM still has (e.g. **#3**).

**XGB target:** match LightGBM on **#1, #2, #4** first; **#3** is a shared CPU gap affecting both libraries.

---

## Upstream issues

| Issue | What it tracks |
|---|---|
| [#2662](https://github.com/shap/shap/issues/2662) | Original pandas categorical XGBoost (2022, open/stale). **#3462** fixed UBJSON load; no-bg shortcut works. |
| [#5068](https://github.com/shap/shap/issues/5068) | ENH: enable XGB categorical via LightGBM-style bitmask (Jul 2026). **Closest match for this work.** |
| [#4182](https://github.com/shap/shap/issues/4182) | GPU tracker — categorical xfail item; XGB still partial after **#4997**. |

No separate issues for “background” vs “interventional” — both are sub-gaps of the above.

---

## Evolution arc (LightGBM → XGB)

| Step | LightGBM | XGBoost |
|---|---|---|
| 1. Python plumbing | **#4171** ✅ | **Branch WIP** — `XGBTreeModelLoader` + `SingleTree` |
| 2. CPU `_cext` | **#4171** ✅ | **Done** (shared) — no XGB-specific C++ work |
| 3. GPU `_cext_gpu` | **#5020** ✅ | **Done** (shared) — `CategoryConstraint` is model-agnostic |
| 4. Tests | **#4171** additivity; **#5026** CPU/GPU synthetic | **Not done** |

**#4171 GPU:** warning only until **#5020**.

---

## Active gaps (summary)

See **Full case matrix** above. Short version for XGB before this branch:

| Path | Engine | Was broken how |
|---|---|---|
| CPU `tree_path_dependent` + **background** (#2) | `_cext` | `NotImplementedError` (#4997) + missing plumbing |
| CPU **interventional** (#3) | `_cext` | guard + `tree_shap_indep` has no categorical routing |
| **GPUTreeExplainer** (#4–6) | `_cext_gpu` | guard + missing plumbing |

**Not broken:** CPU #1 — `pred_contribs` shortcut.

---

## Implementation order

1. **XGB #2:** CPU `tree_path_dependent` + background — plumbing + guard relax ✅
2. **XGB #3:** CPU interventional — `tree_shap_indep` in `tree_shap.h` (shared LGBM gap)
3. **XGB #4–6:** GPU — validate on CUDA; **#5** = `assert_gpu_matches_cpu` vs #2

One `_tree.py` diff serves all paths; land tests per path.

---

## Validation strategy (corrected)

**Do not** compare background vs no-background SHAP values — different algorithms.

| Path | Oracle |
|---|---|
| No-bg `tree_path_dependent` | Optional: `pred_contribs` vs `_cext` (same definition) |
| Bg `tree_path_dependent` | Additivity + (later) GPU==CPU |
| Interventional | Additivity + GPU==CPU only |

**LightGBM precedent (#4171/#5020/#5026):** no hardcoded golden vectors; additivity on real models + synthetic CPU/GPU parity + maintainer GPU review.

---

## Code changes (fork)

### CPU `_cext` (`tree_shap.h`)

- `threshold_types`: `0` = numeric, `1` = LightGBM cat (`2^(cat-1)`, in-set → left), **`2` = XGB cat** (`2^cat`, in-set → right).
- `category_in_threshold_xgb()` + `tree_split_child()` shared by predict + Tree SHAP.

### Plumbing (`_tree.py`)

1. `XGBTreeModelLoader`: cat nodes → `threshold_types=2`, `thresholds=sum(2**cat)`.
2. `_convert_xgboost_categorical_array()`: pandas `category` columns → `cat.codes` before `_cext` (not raw `.values`).
3. `_xgboost_cat_unsupported`: allow when trees have `threshold_types==2`; interventional still raises.

### GPU (`_cext_gpu.cu`)

- Type-2 branch direction mirrored (not hardware-validated in this PR).

### Tests

- `tests/explainers/test_tree.py` — `TestExplainerXGBoostCategorical`: additivity vs `output_margin` (#2).
- `tests/explainers/test_gpu_tree.py` — smoke test present; skipped without CUDA.

---

## Not in this PR

| Item | Status |
|---|---|
| **Interventional + categoricals** (#3) | Still `NotImplementedError`; needs `tree_shap_indep` cat routing (shared LGBM gap) |
| **GPU end-to-end** (#4–6) | Code touched; not proven on CUDA hardware |
| **Per-feature oracle vs `pred_contribs`** | Not tested — only row-level additivity vs `output_margin` |
| CatBoost categorical, upstream PR | Out of scope |

---

## Root cause (XGB #2)

Two bugs, not child-swap:

1. **Split semantics** — XGB uses `2^cat` bitmask; in-set → **right**, not-in-set → left (not LightGBM `2^(cat-1)` in-set → left).
2. **Input encoding** — trees split on **pandas category codes**; SHAP was passing raw category values from `DataFrame.values`.

---

## Risk

Explain-time category integers must be **codes** (0…k-1), aligned with training. Raw label values (e.g. `Workclass=4.0` with code `3`) route wrong.

---

## Repro: LGBM #3 interventional + background (upstream, no fork changes)

Run from repo root with `lightgbm` installed (`pip install lightgbm`).

### Numeric-only (passes — not the bug)

```python
import lightgbm
import shap
import numpy as np

X, y = shap.datasets.california(n_points=2000)
Xe, Xb = X.iloc[:10], X.iloc[100:200]
model = lightgbm.LGBMRegressor(n_estimators=50).fit(X, y)
preds = model.predict(Xe, raw_score=True)

ex = shap.TreeExplainer(model, Xb, feature_perturbation="interventional")
sv = ex.shap_values(Xe, check_additivity=True)
err = np.max(np.abs(sv.sum(axis=1) + ex.expected_value - preds))
print("numeric interventional max err:", err)  # ~1e-7
```

### Categorical splits (fails — cat SHAP = 0)

```python
import lightgbm
import shap
import numpy as np
import pandas as pd

X, y = shap.datasets.california(n_points=2000)
X = X.copy()
X["HouseAgeGroup"] = pd.cut(
    X["HouseAge"], bins=[-np.inf, 17, 27, 37, np.inf], labels=[0, 1, 2, 3], right=False
).astype(int)
cat_idx = X.columns.get_loc("HouseAgeGroup")

model = lightgbm.LGBMRegressor(n_estimators=50, max_cat_to_onehot=1)
model.fit(X, y, categorical_feature=[cat_idx])
Xe, Xb = X.iloc[:10], X.iloc[100:200]
preds = model.predict(Xe, raw_score=True)

# #3 — broken (use check_additivity=False to inspect; True raises ExplainerError)
ex = shap.TreeExplainer(model, Xb, feature_perturbation="interventional")
sv = ex.shap_values(Xe, check_additivity=False)
gap = np.abs(sv.sum(axis=1) + ex.expected_value - preds)
print("cat interventional per-row gap:", gap)
print("cat interventional max err:", gap.max())   # ~0.039 worst row; some rows ~0.003
print("HouseAgeGroup SHAP:", sv[:, cat_idx])       # all zeros

# #2 — control
ex2 = shap.TreeExplainer(model, Xb, feature_perturbation="tree_path_dependent")
sv2 = ex2.shap_values(Xe, check_additivity=True)
print("cat tree_path_dep max err:", np.max(np.abs(sv2.sum(axis=1) + ex2.expected_value - preds)))  # ~1e-7
print("HouseAgeGroup SHAP:", sv2[:, cat_idx])      # non-zero
```

**Expected (Mar 2026 runs):** numeric max err ~`1e-7`. Categorical: `check_additivity=True` **throws** `ExplainerError` (e.g. sum 2.518 vs pred 2.480 on one row). With `check_additivity=False`, `max(|sum(shap)+expected−pred|)` ~**0.039** on worst row (row 6); other rows ~0.003. `HouseAgeGroup` SHAP all `0.0` under interventional; tree_path_dependent control ~`1e-7` with non-zero cat SHAP.

**Root cause:** `tree_shap_indep` in `tree_shap.h` — numeric `x > thres` only; ignores `threshold_types == 1`.

---

## References

- [#4171](https://github.com/shap/shap/pull/4171) — LightGBM plumbing + CPU `_cext`
- [#5020](https://github.com/shap/shap/pull/5020) — GPU categorical kernel
- [#5026](https://github.com/shap/shap/pull/5026) — synthetic CPU/GPU equivalence
- [#4997](https://github.com/shap/shap/pull/4997) — `_xgboost_cat_unsupported` guard
- [#3462](https://github.com/shap/shap/pull/3462) — UBJSON model load
