# SHAP theory notes

---

# Shapley theory

## The problem

You have a model prediction and you want to know: **how much did each feature contribute?**

Shapley's rule for one feature:

1. List **every feature set that does not include it** → call each set **S**
2. For each set, ask: **if I turn this feature on, how much does the score go up?** → the **bump**
3. **Average** those bumps → that's the feature's fair share → **φ** (phi)

Do that for every feature.

## Why 2 to the power of n

Each feature is either **on** or **off**. Two choices per feature → **2ⁿ** combos total.

- 2 features → 4 combos
- Each feature is checked against **2^(n−1)** sets that exclude it → 2 sets when n = 2

You need a score for **every** combo — written **v(S)**. That's the expensive part.

**v(S)** = score when exactly the features in S are on.  
**∅** = empty set = nothing on.

### Time complexity — exact Shapley

**O(2ⁿ)** in the number of features **n** — one **v(S)** per combo.  
20 features → ~1 million combos. Each combo may need a full model prediction → impractical for large **n**.

## Worked example — feature A (with B in the mix)


| What's on | S      | A   | B   | Score | v(S) |
| --------- | ------ | --- | --- | ----- | ---- |
| nothing   | ∅      | off | off | 0     | 0    |
| A only    | {A}    | on  | off | 3     | 3    |
| B only    | {B}    | off | on  | 5     | 5    |
| both      | {A, B} | on  | on  | 10    | 10   |


Every set **without A** — average what A adds:


| Set without A | S   | Score (A off) | v(S) | Score (A on) | v(S ∪ {A}) | bump           |
| ------------- | --- | ------------- | ---- | ------------ | ---------- | -------------- |
| nothing       | ∅   | 0             | 0    | 3            | 3          | 3 − 0 = **3**  |
| B only        | {B} | 5             | 5    | 10           | 10         | 10 − 5 = **5** |


In notation the bumps are: v({A}) − v(∅) = 3 and v({A,B}) − v({B}) = 5

**A's fair share → φ_A = (3 + 5) / 2 = 4**

Same steps for every other feature. All shares must add up to the full prediction minus baseline:

both on minus nothing → v({A,B}) − v(∅) = 10 − 0 = **10**  
or in notation: **φ_A + φ_B = 10**

With 2 features every bump has weight **½** — plain average.

## ML model — house price example (2 features)

The toy example had features literally **on** or **off**. A real model always needs a full input row. So the question becomes: **when a feature is "off", what value does the model see?**

You never set sqft to 0 or delete a column. "Off" means **not fixed to this house** — the value is filled from a **background dataset** (sample of training houses).

Stick to **2 features** — sqft and bedrooms — same as A and B above.

### Setup

**Background average house** (stands in for any feature that is "off"):


| Feature  | Avg value |
| -------- | --------- |
| sqft     | 1500      |
| bedrooms | 3         |


**This house** (the row we explain):


| Feature  | Value |
| -------- | ----- |
| sqft     | 2000  |
| bedrooms | 4     |


In real SHAP, background is many rows and we average predictions. Here we use **one average row** to keep the numbers simple.

### What v(S) means now

**S** = which features stay fixed to **this house**. Everything else uses the **background average**.


| What's on     | S                | Model input (sqft, bed) | v(S) |
| ------------- | ---------------- | ----------------------- | ---- |
| nothing       | ∅                | 1500, 3 (avg)           | 300  |
| sqft only     | {sqft}           | 2000, 3                 | 340  |
| bedrooms only | {bedrooms}       | 1500, 4                 | 320  |
| both          | {sqft, bedrooms} | 2000, 4                 | 400  |


**v(S)** = `model.predict(...)` for that row. (With many background rows, average those predictions instead.)

### φ_sqft — every set without sqft


| Set without sqft | S          | v(S) | v(S ∪ {sqft}) | bump               |
| ---------------- | ---------- | ---- | ------------- | ------------------ |
| nothing          | ∅          | 300  | 340           | 340 − 300 = **40** |
| bedrooms only    | {bedrooms} | 320  | 400           | 400 − 320 = **80** |


**φ_sqft = (40 + 80) / 2 = 60**

### φ_bedrooms — every set without bedrooms


| Set without bedrooms | S      | v(S) | v(S ∪ {bedrooms}) | bump               |
| -------------------- | ------ | ---- | ----------------- | ------------------ |
| nothing              | ∅      | 300  | 320               | 320 − 300 = **20** |
| sqft only            | {sqft} | 340  | 400               | 400 − 340 = **60** |


**φ_bedrooms = (20 + 60) / 2 = 40**

**expected_value** ≈ **v(∅) = 300**

**Additivity:** 300 + 60 + 40 = **400** = prediction for this house ✓

---

# Tree SHAP

Exact Shapley loops over **2ⁿ** combos (4 in the house example). **Tree SHAP** computes the same **φ** values without that loop by walking the tree:

- Feature **on** → use this house's value at each split
- Feature **off** → split the path: weight left/right children by how many background houses went that way

"Off" still means **not committed to this house's value** — not zero, not missing.

## Why Tree SHAP is cheap

Exact Shapley treats the model as a **black box** — score every on/off combo → **O(2ⁿ)**. Tree SHAP opens the box and uses three optimizations.

### 1. Coalitions → tree walks

**Exact Shapley** builds arbitrary combos: {sqft}, {bedrooms}, {both}, {nothing}. The model must be scored on each — even though a tree never works that way.

**A tree only asks one question per node**, in a fixed order. Example:

```
root: sqft ≤ 1800?
├─ yes → bedrooms ≤ 3?
│         ├─ yes → leaf $280k
│         └─ no  → leaf $350k
└─ no  → leaf $400k
```

To predict **this house** (2000 sqft, 4 bed): walk **3 steps** → one leaf. You never build a fake row with "sqft on, bedrooms off" as a separate `predict` call — the tree structure already defines how features combine.

**Tree SHAP does the same kind of walk** to hand out credit. It recurses through these nodes and accumulates each feature's **φ** along the way. Work ≈ **steps in the tree**, not **2ⁿ coalitions**.

### 2. "Feature off" = merge subtrees

Exact Shapley with bedrooms **off** still needs a bedroom value for every coalition (from background). That sounds like many different inputs.

At the node `bedrooms ≤ 3?`, Tree SHAP asks a different question: **"we haven't fixed bedrooms for this house yet — what would the subtree outputs be on average?"**

- **Left child** (≤3 bed): average leaf value over background houses that went left, weighted by count
- **Right child** (>3 bed): same for houses that went right
- **Merge**: weighted sum of the two children — one number for "bedrooms still unknown"

So one split is handled by **combining two subtrees**, not by scoring 2ⁿ bedroom combos. When bedrooms flips to **on**, you follow **this house's** path (4 bed → right). When **off**, you keep the merged mix.

That's how "feature off" avoids exponential work: **uncertainty = blend children**, not enumerate all coalitions.

### 3. Background counted once

Naive Shapley + background: for every coalition, loop all 100 background houses, build a row, `predict`, average → huge.

**Tree SHAP pre-counts at init:**

| Node | Question | background houses that reached here |
|------|----------|-------------------------------------|
| root | sqft ≤ 1800? | 100 |
| left | bedrooms ≤ 3? | 60 went left, 40 went right |
| leaf | — | e.g. 25 houses in this leaf |

Those counts are **`node_sample_weight`** — stored on the tree. Explaining this house (2000 sqft, 4 bed) **reuses** them; it never re-loops 100 houses at every coalition.

Init: count background once per node. Explain: walk tree + merge with stored weights.

### Toy contrast (sqft + bedrooms)


| Approach      | Work                                                                            |
| ------------- | ------------------------------------------------------------------------------- |
| Exact Shapley | 4 coalitions × full `predict`                                                   |
| Tree SHAP     | Walk each tree (~depth steps); at each split, combine two children with weights |


Same **φ** (path-dependent), but work scales with **tree size**, not **2ⁿ**.

### Time complexity — Tree SHAP

**Polynomial** in tree size — **not** O(2ⁿ). Per row explained, roughly **O(T · L · D²)**:


| Symbol | Meaning             |
| ------ | ------------------- |
| **T**  | number of trees     |
| **L**  | max leaves per tree |
| **D**  | max depth           |


Same Shapley **φ** values (for tree_path_dependent), feasible on models with hundreds of features. Applies **with or without** background — background weights are pre-aggregated at init, not a 2ⁿ loop per row.

## In code

```python
explainer = shap.TreeExplainer(
    model,
    background_houses,
    feature_perturbation="tree_path_dependent",
)
phi = explainer.shap_values(this_house)  # [φ_sqft, φ_bedrooms]
```

Default `TreeExplainer(model, background)` uses **interventional** Shapley — same coalition idea, different rule for filling "off" features (each filled independently from background). **tree_path_dependent** matches the path-walking description above.

## Execution path — init vs `shap_values`

### `TreeExplainer(..., background)` — init

- Python builds `TreeEnsemble` with background rows
- Each tree calls **`dense_tree_update_weights`** → walks every background row, increments **`node_sample_weight[i]`** at each node it hits
- **Not feature averages** — **counts** (how many background houses went left/right at each split)
- Also computes **`expected_value`** ≈ mean prediction over background (**v(∅)**)

### `explainer.shap_values(this_house)` — explain

- **`dense_tree_shap`** → **`dense_tree_path_dependent`**
- For each row × each tree → **`tree_shap`** → **`tree_shap_recursive`**
- Uses **pre-stored** `node_sample_weights` for hot/cold fractions — **no background loop here**
- Optionally **`tree_predict`** for additivity check

**Summary:** init = pre-count background through the tree; `shap_values` = explain this house using those counts.
