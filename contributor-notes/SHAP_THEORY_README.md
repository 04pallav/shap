# SHAP TreeExplainer — theory notes

Personal reference for how `TreeExplainer` chooses algorithms, what “additivity” means, and how categorical splits are routed. Implementation tracker: [XGB_CATEGORICAL_FIX.md](./XGB_CATEGORICAL_FIX.md).

---

## 1. What TreeExplainer computes

For tree models, SHAP values attribute each feature’s contribution to the model output (margin / log-odds when `model_output="raw"`). For a single row:

```
model_output(x) ≈ expected_value + Σᵢ shapᵢ(x)
```

`expected_value` is the mean model output over the background set (when background is provided) or derived from tree structure (no background).

**Additivity** is this identity holding row-by-row. SHAP enforces it in `assert_additivity` by comparing `sum(shap) + expected_value` to **`TreeEnsemble.predict(X)`** — SHAP’s own C++ tree walk — not necessarily the booster library’s native `predict`.

---

## 2. The two axes that matter

Every `TreeExplainer` call sits at a point in this grid:

| Axis | Options | Default |
|------|---------|---------|
| **Background** | None vs provided `data` | None |
| **Perturbation** | `tree_path_dependent` vs `interventional` | `auto` |

### `feature_perturbation="auto"` (`_tree.py:294–295`)

```python
feature_perturbation = "interventional" if self.data is not None else "tree_path_dependent"
```

- **No background** → `tree_path_dependent`
- **With background** → `interventional`

To hit path-dependent SHAP with a background (case **#2** below), pass **`feature_perturbation="tree_path_dependent"` explicitly**.

### Perturbation semantics (informal)

| Mode | Idea | Needs background? |
|------|------|-------------------|
| **`tree_path_dependent`** | Feature is present or absent along the path the tree actually took; conditional expectations follow tree structure | No (but can use one) |
| **`interventional`** | Break correlations: marginalize each feature over the background distribution independently | Yes (warns / falls back if missing) |

These are **different algorithms**. SHAP values from #1 and #2 are not comparable even on the same model.

---

## 3. The six-path matrix

Grid: **engine × perturbation × background**. Omitted cell: `interventional` without background — SHAP emits `FutureWarning` and silently switches to `tree_path_dependent` (same as #1 / #4).

### Case numbers (same for every library)

| # | Engine | Perturbation | Background | Typical code path |
|---|--------|--------------|------------|-------------------|
| 1 | CPU | `tree_path_dependent` | No | Native `pred_contribs` shortcut |
| 2 | CPU | `tree_path_dependent` | Yes | `_cext` (`dense_tree_shap`) |
| 3 | CPU | `interventional` | Yes | `_cext` (`tree_shap_indep`) |
| 4 | GPU | `tree_path_dependent` | No | Booster GPU contrib or `_cext_gpu` |
| 5 | GPU | `tree_path_dependent` | Yes | `_cext_gpu` |
| 6 | GPU | `interventional` | Yes | `_cext_gpu` interventional |

### Shortcut gate (`_tree.py:715–724`)

`pred_contribs` / `pred_contrib` shortcut runs only when **all** of:

1. `feature_perturbation == "tree_path_dependent"`
2. `model_type` is xgboost / lightgbm / catboost (not `internal`)
3. **`self.data is None`** (no background)

Any background forces `_cext` for path-dependent mode.

---

## 4. Code paths (CPU)

```
TreeExplainer.shap_values(X)
│
├─ #1  tree_path_dependent + no background
│      └─ _short_circuit_tree_path_dependent_to_external_shap_calculation()
│         ├─ XGBoost: booster.predict(..., pred_contribs=True)
│         ├─ LightGBM: booster.predict(..., pred_contrib=True)
│         └─ CatBoost: get_feature_importance(..., ShapValues)
│
├─ #2  tree_path_dependent + background
│      └─ _cext.dense_tree_shap  (tree_shap_recursive in tree_shap.h)
│
└─ #3  interventional + background
       └─ _cext.dense_tree_shap  (tree_shap_indep in tree_shap.h)
```

GPU mirror: `GPUTreeExplainer` → `_cext_gpu` for #4–6.

---

## 5. Categorical split routing (`tree_shap.h`)

`threshold_types` on each split node:

| Type | Library | Bitmask | In-set goes |
|------|---------|---------|-------------|
| `0` | all | numeric threshold | left if `x ≤ thres` |
| `1` | LightGBM | `2^(cat−1)` | **left** |
| `2` | XGBoost | `2^cat` (0-based code) | **right** |

Shared entry point: `tree_split_child()` — used by both **predict** and **SHAP** in `_cext`.

**Input encoding:** trees split on integer category **codes** (0…k−1), not arbitrary label values. Mismatch between training codes and explain-time values → wrong leaf → broken SHAP.

### Known LGBM edge case (cat code 0)

LightGBM loader builds bitmask with `2 ** (cat - 1)`. Category **0** gives `2^(-1)` — invalid / wrong in C++ (`1 << -1`). Empirically: `TreeEnsemble.predict` can diverge from native LightGBM `predict(raw_score=True)` on rows with cat=0, while **internal** additivity (`check_additivity=True` vs `TreeEnsemble.predict`) still passes. No-bg path avoids this by using LightGBM’s own `pred_contrib`.

XGB type-2 uses **0-based** `2^cat` — aligned with `enable_categorical` + `cat.codes`.

---

## 6. Validation — which oracle to use

**Do not** compare SHAP values across different cases (#1 vs #2, or path-dependent vs interventional). Different definitions.

| Case | Valid oracle | Invalid oracle |
|------|--------------|----------------|
| #1 no-bg path-dependent | Native `pred_contribs` (optional cross-check) | Background path SHAP |
| #2 bg path-dependent | `check_additivity=True` (vs `TreeEnsemble.predict`) | Native booster predict if `TreeEnsemble.predict` diverges |
| #3 interventional | `check_additivity=True` | `pred_contribs` |
| #5 GPU path-dependent | `assert_gpu_matches_cpu` vs CPU #2 | — |

**Per-feature correctness** (golden SHAP per feature) has no cheap external oracle on #2/#3 — only row-level additivity + (for GPU) CPU parity. Brute-force tree SHAP exists in tests for tiny synthetic numeric trees only.

### LightGBM #3 failure mode (upstream, pre-fix)

`tree_shap_indep` uses numeric `x > threshold` only — ignores `threshold_types == 1`. Symptom: categorical feature SHAP = 0, additivity fails vs model output. Confirms #3 needs C++ work separate from categorical plumbing in #2.

---

## 7. Library status snapshot (Mar 2026)

### LightGBM categorical (#4171 / #5020)

| # | Status | Notes |
|---|--------|-------|
| 1 | ✅ Works | `pred_contrib` shortcut |
| 2 | ✅ Works | `_cext` path-dependent; `check_additivity=True` passes |
| 3 | ❌ Broken | `tree_shap_indep` no cat routing; cat SHAP = 0 |
| 4–6 | GPU track | See #5020 / #5026 |

### XGBoost categorical (`enable_categorical=True`)

| # | Status | Notes |
|---|--------|-------|
| 1 | ✅ Works | `pred_contribs` since XGB 1.7+ |
| 2 | 🚧 PR [#5107](https://github.com/shap/shap/pull/5107) | `threshold_types=2`, cat codes, additivity vs `output_margin` |
| 3 | ❌ Not in PR | Same `tree_shap_indep` gap as LGBM |
| 4–6 | ❌ Not in PR | `_cext_gpu` touched, not CUDA-validated |

---

## 8. Minimal API recipes

```python
# Case #1 — no background, path-dependent (default)
explainer = shap.TreeExplainer(model)
# or explicitly:
explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")

# Case #2 — background + path-dependent (_cext) — NOT default
explainer = shap.TreeExplainer(
    model, background, feature_perturbation="tree_path_dependent"
)

# Case #3 — background + interventional (default when background given)
explainer = shap.TreeExplainer(model, background)
# or:
explainer = shap.TreeExplainer(model, background, feature_perturbation="interventional")
```

---

## 9. Related docs

| Doc | Purpose |
|-----|---------|
| [XGB_CATEGORICAL_FIX.md](./XGB_CATEGORICAL_FIX.md) | XGB implementation tracker, repro scripts, issue map |
| [OPEN_PRS.md](./OPEN_PRS.md) | Fork / upstream PR status |

---

## 10. Key source locations

| Topic | File | Lines (approx) |
|-------|------|----------------|
| `auto` perturbation | `shap/explainers/_tree.py` | 294–295 |
| Shortcut gate | `shap/explainers/_tree.py` | 715–724 |
| Additivity check | `shap/explainers/_tree.py` | 779–780, 955–973 |
| LGBM cat loader | `shap/explainers/_tree.py` | 2052–2061 |
| Cat routing C++ | `shap/cext/tree_shap.h` | 181–211 |
| XGB cat guard | `shap/explainers/_tree.py` | 112–149 |
