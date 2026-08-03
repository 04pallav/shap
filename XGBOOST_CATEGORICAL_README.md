# XGBoost categorical CPU TreeExplainer — PR code walkthrough

Branch: `support/xgboost-categorical-cpu`
PR: https://github.com/04pallav/shap/pull/3
Files: <code>shap/cext/tree_shap.h</code>, <code>shap/explainers/_tree.py</code>, <code>tests/explainers/test_tree.py</code>

Each section shows before (`master`) and after (this PR) code with inline comments on every changed line.

Label convention: `Before (master):` / `After (this PR):` — plain text, no bold.

## Review tracker

Mark `[x]` when you’ve read the section and understand the change. Update as you go.

| ID | Section | File | Reviewed |
|----|---------|------|----------|
| 1a | LightGBM helper comment (logic unchanged) | tree_shap.h | [ ] |
| 1b | NEW — XGBoost category bitmask (category_in_threshold_xgb) | tree_shap.h | [ ] |
| 1c | NEW — unified split dispatcher (tree_split_child; tfloat fix) | tree_shap.h | [ ] |
| 1d | tree_predict calls dispatcher | tree_shap.h | [ ] |
| 1e | tree_update_weights calls dispatcher | tree_shap.h | [ ] |
| 1f | tree_shap_recursive calls dispatcher | tree_shap.h | [ ] |
| 2a | New imports (pd, npt) | tree.py | [x] |
| 2b | xgboost_cat_unsupported guard | tree.py | [ ] |
| 2c | TreeExplainer init — sync encoded background | tree.py | [x] |
| 2d | validate_inputs — explain-path encoding | tree.py | [ ] |
| 2e | shap_values passes perturbation mode | tree.py | [ ] |
| 2f | TreeEnsemble.input_transform field | tree.py | [ ] |
| 2g | NEW — to_input_array | tree.py | [ ] |
| 2h | set_xgboost_model_attributes hook + background | tree.py | [ ] |
| 2i | TreeEnsemble.predict same input gate | tree.py | [ ] |
| 2j | SingleTree reads threshold_type from dict | tree.py | [ ] |
| 2k | NEW — XGBTreeModelLoader.transform_input | tree.py | [ ] |
| 2l | XGBTreeModelLoader tags cat nodes type=2 | tree.py | [ ] |
| 3a | Test fixture (XGB + categoricals) | test_tree.py | [ ] |
| 3b | Additivity vs output_margin | test_tree.py | [ ] |
| 3c | Interventional mode raises | test_tree.py | [ ] |

---

## 1. `shap/cext/tree_shap.h` — split semantics in C++

### 1a. LightGBM helper comment (unchanged logic, new comment)

Before (master):
```cpp
inline bool category_in_threshold(float threshold, float category) {
    int category_flag = (1 << (int(category) - 1));   // 1-based level → bit (cat-1)
    return (int(threshold) & category_flag) != 0;     // in bitmask → true
}
```

After (`tree_shap.h:181-185`):
```cpp
// LightGBM-style categorical splits (threshold_types == 1): 1-based category levels.
inline bool category_in_threshold(tfloat threshold, tfloat category) {
    int category_flag = (1 << (int(category) - 1));   // SAME: LGBM uses 1-based levels
    return (int(threshold) & category_flag) != 0;     // SAME: test bitmask
}
```
> Comment only. Behaviour for `threshold_types == 1` is unchanged.

---

### 1b. NEW — XGBoost category bitmask (`tree_shap.h:187-195`)

Before (master): did not exist. XGB cat nodes were either un-tagged (type 0) or mis-handled as type 1.

After (this PR):
```cpp
// XGBoost-style categorical splits (threshold_types == 2): 0-based category codes.
inline bool category_in_threshold_xgb(tfloat threshold, tfloat category) {
    const int category_int = static_cast<int>(category);  // feature value is a code, not a label
    if (category_int < 0) {                               // missing / invalid code (like pandas -1)
        return false;                                     // → not in the split set
    }
    const int category_flag = 1 << category_int;          // XGB bitmask: 2^code (0-based)
    return (static_cast<int>(threshold) & category_flag) != 0;
}
```
> Why: XGBoost stores category codes (0,1,2…) in the bitmask as `2^code`, not `2^(level-1)` like LightGBM.

---

### 1c. NEW — unified split dispatcher (`tree_shap.h:197-211`)

Before (master): same logic copy-pasted in 3 functions (`tree_predict`, `tree_update_weights`, `tree_shap_recursive`):
```cpp
} else if (trees.threshold_types[pos] == 0 && x[feature] <= trees.thresholds[pos]) {
    node = trees.children_left[pos];                    // numeric: ≤ threshold → left
} else if (trees.threshold_types[pos] == 1 && category_in_threshold(...)) {
    node = trees.children_left[pos];                    // LGBM cat: in-set → left
} else {
    node = trees.children_right[pos];                   // everything else → right
}
// No branch for XGB cat (type 2). Cat nodes fell through to → right incorrectly.
```

After (this PR):
```cpp
inline unsigned tree_split_child(
    int split_type, tfloat threshold, tfloat feature_value,
    unsigned left_child, unsigned right_child
) {
    if (split_type == 0) {
        return feature_value <= threshold ? left_child : right_child;  // SAME as before
    }
    if (split_type == 1) {
        return category_in_threshold(threshold, feature_value)
            ? left_child : right_child;                                // SAME: LGBM in-set → left
    }
    if (split_type == 2) {
        // XGBoost: categories in the bitmask go right; missing/other categories go left.
        return category_in_threshold_xgb(threshold, feature_value)
            ? right_child : left_child;                              // NEW: XGB in-set → RIGHT
    }
    return right_child;                                                // SAME fallback as old else
}
```
> Why: XGB `Decision()` sends in-set categories to the right child (opposite of LightGBM).
> Params must be `tfloat` (double), not `float` — master compared thresholds inline as doubles; `float` downcast broke near-tie numeric splits (e.g. iris `test_summary_bar_multiclass`).

---

### 1d. `tree_predict` — call dispatcher (`tree_shap.h:225-233`)

Before (master):
```cpp
if (x_missing[feature]) {
    node = trees.children_default[pos];
} else if (trees.threshold_types[pos] == 0 && x[feature] <= trees.thresholds[pos]) {
    node = trees.children_left[pos];
} else if (trees.threshold_types[pos] == 1 && category_in_threshold(trees.thresholds[pos], x[feature])) {
    node = trees.children_left[pos];
} else {
    node = trees.children_right[pos];
}
```

After (this PR):
```cpp
if (x_missing[feature]) {
    node = trees.children_default[pos];               // SAME: missing → default child
} else {
    node = tree_split_child(                          // NEW: one call handles types 0, 1, 2
        trees.threshold_types[pos], trees.thresholds[pos], x[feature],
        trees.children_left[pos], trees.children_right[pos]
    );
}
```
> Types 0 and 1 produce identical child choices as before. Type 2 is new.

---

### 1e. `tree_update_weights` — same refactor (`tree_shap.h:288-296`)

Before (master): duplicated if/else (same as 1d).

After (this PR):
```cpp
if (x_missing[feature]) {
    node = trees.children_default[pos];
} else {
    node = tree_split_child(
        trees.threshold_types[pos], trees.thresholds[pos], x[feature],
        trees.children_left[pos], trees.children_right[pos]
    );
}
```
> Background sample weights must follow the same tree path as prediction.

---

### 1f. `tree_shap_recursive` — SHAP hot-path (`tree_shap.h:477-485`)

Before (master):
```cpp
if (x_missing[split_index]) {
    hot_index = children_default[node_index];
} else if (type == 0 && x[split_index] <= thresholds[node_index]) {
    hot_index = children_left[node_index];
} else if (type == 1 && category_in_threshold(thresholds[node_index], x[split_index])) {
    hot_index = children_left[node_index];
} else {
    hot_index = children_right[node_index];
}
```

After (this PR):
```cpp
if (x_missing[split_index]) {
    hot_index = children_default[node_index];         // SAME
} else {
    hot_index = tree_split_child(                       // SAME dispatcher as predict/weights
        type, thresholds[node_index], x[split_index],
        children_left[node_index], children_right[node_index]
    );
}
```
> This is where wrong XGB cat direction caused SHAP values to not add up to `output_margin`.

---

## 2. `shap/explainers/_tree.py` — Python loader + input encoding

### 2a. New imports (`_tree.py:9-10`)

Before (master): not present.

After (this PR):
```python
from collections.abc import Callable   # type hint for input_transform hook
from functools import partial          # bind cat_feature_indices into transform_input
```

---

### 2b. `_xgboost_cat_unsupported` — guard logic (`_tree.py:112-150`)

Before (master):
```python
def _xgboost_cat_unsupported(model: TreeEnsemble) -> None:
    ...
    if has_cat_columns:
        raise NotImplementedError(                      # ALWAYS blocked _cext for any XGB cat
            "Categorical split is not yet supported. ..."
        )
```

After (this PR):
```python
def _xgboost_cat_unsupported(model: TreeEnsemble, feature_perturbation: str | None = None) -> None:
    ...
    if not has_cat_columns:
        return                                          # NEW: numeric XGB → no-op, proceed

    if feature_perturbation == "interventional":
        raise NotImplementedError(                      # NEW: explicit interventional block
            "Categorical split is not yet supported with feature_perturbation="
            '"interventional". You can use feature_perturbation="tree_path_dependent".'
        )

    threshold_types = getattr(model, "threshold_types", None)
    if threshold_types is None or not np.any(threshold_types == 2):
        raise NotImplementedError(                      # Only block if loader didn't tag type-2 nodes
            "Categorical split is not yet supported. ..."
        )
    # If we reach here: XGB cat + tree_path_dependent + type-2 nodes → _cext allowed
```
> Before (master): any categorical column → `_cext` blocked even when we could fix it.
> After (this PR): allow `_cext` when loader set `threshold_types == 2`; still block interventional.

---

### 2c. `TreeExplainer.__init__` — background data (`_tree.py:289-332`)

#### Why two objects both store background

`TreeExplainer` and `TreeEnsemble` each have their own `data` / `data_missing` fields. That is existing SHAP design — not introduced by this PR. What matters is who reads which copy when `_cext` runs:

```python
# shap_values → dense_tree_shap (simplified)
_cext.dense_tree_shap(
    ...,           # trees come from self.model
    X,             # rows to explain — encoded in _validate_inputs via _to_input_array
    X_missing,
    ...,
    self.data,     # ← background: read from TreeExplainer, NOT self.model.data
    self.data_missing,
    ...
)
```

So even after `TreeEnsemble` converts background internally, `TreeExplainer.self.data` must be updated or `_cext` still sees the old values.

#### Init order (why a sync step is required)

```
1. self.data = user DataFrame          # e.g. Workclass column has labels "Private", "Self-emp", …
2. self.data_missing = pd.isna(self.data)   # missing mask on raw DataFrame
3. TreeEnsemble(..., self.data, ...)   # inside: _set_xgboost_model_attributes runs
       → loader sees cat_feature_indices
       → self.model.data = encoded ndarray   # Workclass now 0, 1, 2, … (cat.codes)
       → self.model.data_missing = np.isnan(encoded)
4. WITHOUT sync: TreeExplainer.self.data still step-1 DataFrame
   WITH sync:    TreeExplainer.self.data = step-3 ndarray  ← the two new lines
```

Encoding cannot happen in step 1 because `TreeExplainer` does not yet know if the model is XGB with categoricals — that is only known after `TreeEnsemble` loads the booster and `XGBTreeModelLoader` reads `feature_types`.

#### Concrete example

| Stage | `Workclass` value in background row 0 |
|-------|----------------------------------------|
| User passes DataFrame | `"Private"` (label) or stored code as float `4.0` |
| After `TreeEnsemble._to_input_array` | `3` (pandas `cat.codes` — what the tree split uses) |
| `_cext` with wrong background | compares tree bitmask against `4.0` → wrong leaf → SHAP does not add up |

#### The two sync lines explained

Before (master):
```python
if isinstance(data, pd.DataFrame):
    self.data = data.values                          # BUG: raw category labels (e.g. "Private")
elif isinstance(data, DenseData):
    self.data = data.data
else:
    self.data = data
...
self.model = TreeEnsemble(model, self.data, self.data_missing, model_output)
# self.data never updated after TreeEnsemble conversion
```

After (this PR):
```python
if isinstance(data, DenseData):
    self.data = data.data                            # SAME
else:
    self.data = data                                 # pass DataFrame through; don't convert here yet
...
self.model = TreeEnsemble(model, self.data, self.data_missing, model_output)

self.data = self.model.data
# Copy the encoded background ndarray from TreeEnsemble back onto TreeExplainer.
# _cext.dense_tree_shap reads self.data (explainer), not self.model.data (ensemble).
# Without this line, background rows still have raw labels while explain rows use codes.

self.data_missing = self.model.data_missing
# Re-copy missing mask computed on the encoded ndarray (np.isnan after conversion).
# The mask from line 320 was built on the pre-conversion DataFrame; for categoricals
# that can disagree with which positions are NaN in the float array _cext expects.
```

After (this PR) — same two lines, also used for:
- `expected_value` background mean: `self.model.predict(self.data, ...)` at `_tree.py:365`
- interventional path background integration

For non-XGB models, `TreeEnsemble` does not change `self.data`, so the sync is a no-op (same reference or equivalent array).

---

### 2d. `_validate_inputs` — explain-path input (`_tree.py:488-510`)

This function runs on every `shap_values(X)` / `shap_interaction_values(X)` call **before** `_cext`. It does not compute SHAP; it turns user `X` into the `(n_samples, n_features)` float matrix the C extension walks.

#### Pipeline (both master and this PR)

```
user X  →  convert to ndarray  →  detect single-row (flat_output)  →  cast dtype  →  X_missing  →  dense_tree_shap(..., X, X_missing, ...)
```

#### What master does at the convert step

Before (master):
```python
if isinstance(X, (pd.Series, pd.DataFrame)):
    X = X.to_numpy(dtype=self.model.input_dtype, na_value=np.nan)
flat_output = False
if len(X.shape) == 1:          # pd.Series → 1D (n_features,)
    flat_output = True
    X = X.reshape(1, X.shape[0])   # _cext requires 2D (1, n_features)
if X.dtype != self.model.input_dtype:
    X = X.astype(self.model.input_dtype)
X_missing = np.isnan(X, dtype=bool)
```

`to_numpy` is a generic dump: fine for numeric columns and nullable dtypes (`Int64`, etc.). For pandas `category` columns on an XGB model it does **not** produce the integer **codes** the booster trained on.

Example — one row in `X_explain`, column `Workclass` (categorical):

| Stage | Value seen by tree / `_cext` |
|-------|------------------------------|
| DataFrame cell | `"Private"` (label) |
| `to_numpy()` | label storage or wrong float — not necessarily code `3` |
| XGB split node | bitmask tested against code `3` |

Explain rows then disagree with tree structure → wrong leaf → SHAP does not add up to `output_margin`.

#### What this PR changes at the convert step

After (this PR):
```python
X = self.model._to_input_array(X)   # see 2g: to_numpy, or transform_input for XGB cats
flat_output = False
if len(X.shape) == 1:
    flat_output = True
    X = X.reshape(1, X.shape[0])
# ... same dtype cast, X_missing, assertions ...
```

Only the first line changes. For XGB with categorical features, `_to_input_array` calls `transform_input` and replaces `CategoricalDtype` columns with `col.cat.codes`. Non-XGB models: still `to_numpy` — behaviour matches master.

#### Why the `flat_output` block is still here (and must stay after `_to_input_array`)

`flat_output` controls **output shape**, not encoding:

- User passes one sample as `pd.Series` → after convert, `X` is 1D `(n_features,)`.
- `flat_output = True` records “user gave a single row.”
- `reshape(1, n_features)` is only for `_cext` (always 2D input).
- Later, `_get_shap_output` / `_get_shap_interactions_output` drop the batch dimension when `flat_output` is True → SHAP shape `(n_features,)` or `(n_features, n_features)`, not `(1, …)`.

`_to_input_array` must return a **1D** array for `pd.Series` (see 2g `.reshape(-1)`). If Series were left as `(1, n_features)` before this block, `len(X.shape) == 1` would be false, `flat_output` would stay false, and interaction tests break.

#### Where this fits in `shap_values`

```
TreeExplainer.shap_values(X_explain)
  → _validate_inputs(X_explain, ...)
       → _to_input_array(X_explain)     # explain rows: labels → codes (XGB only)
       → X_missing, flat_output, tree_limit
  → dense_tree_shap(..., X, X_missing, self.data, self.data_missing, ...)
       # background self.data must also be codes — see 2c sync lines
```

Explain path (`X`) and background (`self.data`) are prepared separately; both must use codes before `_cext` runs.

---

### 2e. `shap_values` — pass perturbation mode (`_tree.py:730`)

Before (master):
```python
_xgboost_cat_unsupported(self.model)
```

After (this PR):
```python
_xgboost_cat_unsupported(self.model, self.feature_perturbation)
```
> Guard needs mode to distinguish interventional vs path-dependent.

---

### 2f. `TreeEnsemble.input_transform` field (`_tree.py:1050`)

Before (master): not present.

After (this PR):
```python
self.input_transform: Callable[..., npt.NDArray[Any]] | None = None
```
> Optional hook; set to `partial(transform_input, ...)` when XGB has categorical features.

---

### 2g. NEW — `_to_input_array` (`_tree.py:1645-1655`)

Before (master): conversion inlined in `_validate_inputs` and `predict` via `to_numpy`.

After (this PR):
```python
def _to_input_array(self, X: npt.NDArray[Any] | pd.Series | pd.DataFrame) -> npt.NDArray[Any]:
    """Convert user input to the ndarray format expected by the C extension."""
    if isinstance(X, pd.Series):
        if self.input_transform is not None:
            # Encode cats on a 1-row DataFrame, then flatten back to 1D
            return self.input_transform(X.to_frame().T, dtype=self.input_dtype).reshape(-1)
        return X.to_numpy(dtype=self.input_dtype, na_value=np.nan)   # non-XGB: same as master
    if isinstance(X, pd.DataFrame):
        if self.input_transform is not None:
            return self.input_transform(X, dtype=self.input_dtype)   # XGB: replace cat cols with codes
        return X.to_numpy(dtype=self.input_dtype, na_value=np.nan)
    return X                                                      # ndarray: passthrough
```
> Why `.reshape(-1)` on Series: if we returned `(1, n)` here, `flat_output` would stay `False` and
> `shap_interaction_values` would return `(1, 8, 8)` instead of `(8, 8)` — broke `test_lightgbm_interaction`.

---

### 2h. `_set_xgboost_model_attributes` — hook + background (`_tree.py:1657-1677`)

Before (master):
```python
def _set_xgboost_model_attributes(self, data: npt.NDArray[Any] | None, ...):
    loader = XGBTreeModelLoader(self.original_model)
    self.trees = loader.get_trees(data=data, data_missing=data_missing)  # raw background values
```

After (this PR):
```python
def _set_xgboost_model_attributes(
    self, data: npt.NDArray[Any] | pd.Series | pd.DataFrame | None, ...  # accept pandas background
):
    loader = XGBTreeModelLoader(self.original_model)
    if loader.cat_feature_indices is not None:
        self.input_transform = partial(                              # bind cat column indices
            XGBTreeModelLoader.transform_input,
            cat_feature_indices=loader.cat_feature_indices,
        )
    if data is not None and isinstance(data, (pd.Series, pd.DataFrame)):
        data = self._to_input_array(data)                            # encode background before get_trees
        data_missing = np.isnan(data, dtype=bool)
        self.data = data
        self.data_missing = data_missing
    self.trees = loader.get_trees(data=data, data_missing=data_missing)
```
> `get_trees` uses background rows to compute `node_sample_weight`; those rows must use category codes.

---

### 2i. `TreeEnsemble.predict` — same input gate (`_tree.py:1774`)

Before (master):
```python
if isinstance(X, (pd.Series, pd.DataFrame)):
    X = X.to_numpy(dtype=self.input_dtype, na_value=np.nan)
```

After (this PR):
```python
X = self._to_input_array(X)
```
> Predict and explain must traverse trees with the same encoded values.

---

### 2j. `SingleTree` — read `threshold_type` from dict (`_tree.py:1938-1941`)

Before (master):
```python
self.threshold_types = np.zeros_like(self.thresholds, dtype=np.int32)  # always numeric
```

After (this PR):
```python
if "threshold_type" in tree:
    self.threshold_types = np.asarray(tree["threshold_type"], dtype=np.int32)  # honour explicit types
else:
    self.threshold_types = np.zeros_like(self.thresholds, dtype=np.int32)      # default: numeric
```
> For hand-built tree dicts that specify split types. LightGBM `tree_structure` path unchanged (sets types inline).

---

### 2k. NEW — `XGBTreeModelLoader.transform_input` (`_tree.py:2310-2332`)

Before (master): did not exist. No encoding step.

After (this PR):
```python
@staticmethod
def transform_input(X, *, cat_feature_indices, dtype=np.float64):
    """Convert pandas categorical columns to integer codes for XGBoost tree traversal."""
    if isinstance(X, pd.Series):
        X = X.to_frame().T                               # Series → single-row DataFrame

    if isinstance(X, pd.DataFrame):
        arr = X.to_numpy(dtype=dtype, na_value=np.nan).copy()  # start with label values
        if cat_feature_indices is not None:
            for idx in cat_feature_indices:
                col = X.iloc[:, int(idx)]
                if isinstance(col.dtype, pd.CategoricalDtype):
                    codes = col.cat.codes.to_numpy(dtype=arr.dtype, copy=True)  # 0,1,2… not labels
                    codes[codes < 0] = np.nan            # pandas missing code → NaN for _cext
                    arr[:, int(idx)] = codes             # overwrite column with codes
        return arr
    return X
```
> Before (master): column might hold `" Private"` (label) or `4.0` (code storage); tree splits on code `3`.
> After (this PR): always passes integer codes XGBoost trained on.

---

### 2l. `XGBTreeModelLoader` — tag categorical nodes (`_tree.py:2495-2501`)

Before (master):
```python
tree_categories = self.parse_categories(...)
self.categories.append(tree_categories)
# thresholds stayed 0.0, threshold_types stayed 0 → C++ treated cat splits as numeric ≤ 0
```

After (this PR):
```python
tree_categories = self.parse_categories(...)
self.categories.append(tree_categories)
for node_id, node_cats in enumerate(tree_categories):
    if node_cats:                                        # non-empty → categorical split node
        threshold_types[node_id] = 2                     # tell C++ to use XGB cat semantics
        threshold = 0.0
        for cat in node_cats:
            threshold += 2 ** int(cat)                   # build XGB bitmask: sum of 2^code
        thresholds[node_id] = threshold
```
> Before (master): cat node thresholds ignored; wrong split direction in C++.
> After (this PR): matches XGBoost model JSON: `categories` list + bitmask threshold + type 2 in `_cext`.

---

## 3. `tests/explainers/test_tree.py` — regression tests

### 3a. Fixture (`test_tree.py:1615-1629`)

Before (master): no XGB categorical fixture.

After (this PR):
```python
def _fit_xgboost_categorical_classifier():
    xgboost = pytest.importorskip("xgboost")
    X, y = shap.datasets.adult(n_points=300)
    X = X.copy()
    X["Workclass"] = X["Workclass"].astype("category")   # native pandas categorical column
    clf = xgboost.XGBClassifier(
        n_estimators=8,
        enable_categorical=True,                         # XGB native cat support
        max_depth=4,
        tree_method="hist",
        random_state=0,
    )
    clf.fit(X, y)
    return clf, X.iloc[:20], X.iloc[20:120]              # explain set, background set
```

---

### 3b. Additivity test (`test_tree.py:1635-1647`)

Before (master): no test for background + `_cext` + XGB categorical.

After (this PR):
```python
def test_tree_path_dependent_background_additivity(self):
    clf, X_explain, X_bg = _fit_xgboost_categorical_classifier()
    explainer = shap.TreeExplainer(
        clf, X_bg, feature_perturbation="tree_path_dependent"  # forces _cext (not pred_contribs)
    )
    assert np.any(explainer.model.threshold_types == 2)      # loader tagged cat nodes
    shap_values = explainer.shap_values(X_explain, check_additivity=True)
    margin = clf.predict(X_explain, output_margin=True)
    np.testing.assert_allclose(
        shap_values.sum(1) + explainer.expected_value,       # SHAP + base = model output
        margin,
        rtol=1e-5, atol=1e-5,
    )
```

---

### 3c. Interventional guard test (`test_tree.py:1649-1654`)

After (this PR):
```python
def test_interventional_raises(self):
    clf, X_explain, X_bg = _fit_xgboost_categorical_classifier()
    explainer = shap.TreeExplainer(clf, X_bg, feature_perturbation="interventional")
    with pytest.raises(NotImplementedError, match="interventional"):
        explainer.shap_values(X_explain)                   # must still fail explicitly
```

---

## 4. Quick reference — what each `threshold_types` value means

| Value | Engine | Bitmask | In-set goes to |
|-------|--------|---------|----------------|
| `0` | numeric | n/a | `≤ threshold` → left |
| `1` | LightGBM | `2^(level-1)` | left |
| `2` | XGBoost | `2^code` | right |

---

## 5. Rebuild after editing `tree_shap.h`

```bash
pip install -e .
```
