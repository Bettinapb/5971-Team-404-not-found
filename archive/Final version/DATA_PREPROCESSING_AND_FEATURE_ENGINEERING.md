# Data Preprocessing And Feature Engineering

## Goal

This document defines a leakage-safe pipeline for the Instacart reorder prediction project after the earlier global feature table approach was discarded.

The new design separates the workflow into:

1. `Global-safe preprocessing`
   Raw data cleaning, task reformulation, and export of reusable base tables.
2. `Fold-wise feature engineering`
   Any operation that learns from data structure or target behavior is rebuilt inside each training fold only.

This design is necessary because grouped recommendation rows from the same user are highly dependent, and global fitting of learned features introduces information leakage.

## Data Access

The raw data are downloaded with:

```python
import kagglehub
path = kagglehub.dataset_download(
    "yasserh/instacart-online-grocery-basket-analysis-dataset"
)
```

This mirrors the existing notebooks in the repository, so the cloud runtime does not need a local raw-data directory checked into the repo.

## Task Reformulation

We keep the supervised next-order prediction formulation:

- for each user, the last order with product-level information is treated as the target order
- all earlier orders for that user are treated as history
- candidate rows are formed at the `(user, product)` level using previously purchased products only
- the binary label is:
  - `1` if the candidate product appears in the user target order
  - `0` otherwise

This creates a realistic reorder problem while keeping the candidate space tractable.

## What The Preprocessing Script Is Allowed To Export

The preprocessing script writes only base artifacts that are safe to share across later folds:

- cleaned order tables
- enriched prior interaction table
- target-order labels
- candidate universe based on prior purchases
- product metadata
- order-level basket summaries
- reusable history-derived raw timestamps and order sequence fields

These artifacts do not leak future fold labels by themselves because they do not fit cross-user models or compute fold-sensitive target statistics.

## What Must Be Built Inside Each Training Fold

The following must never be precomputed on the full dataset before evaluation:

- user-level aggregates used by the model
- product-level aggregates used by the model
- user-product aggregates used by the model
- KMeans user clusters
- cluster dummy variables
- cluster-product target rates
- Apriori rules and Apriori rule hits
- ALS user/item factors and ALS-derived similarity features

In the notebook, every outer training fold rebuilds these objects from training users only, and then applies them to the held-out users.

## Global-Safe Preprocessing Steps

### 1. Raw table loading

Load:

- `orders.csv`
- `order_products__prior.csv`
- `order_products__train.csv`
- `products.csv`
- `aisles.csv`
- `departments.csv`

### 2. Supervised cohort restriction

The original Kaggle `train/test` split is not the split we use for final supervised modeling.

Instead, before building the reusable bundle, we keep users whose `orders.eval_set` is either:

- `train`
- `test`

For these users:

- for original `train` users, the `train` order is the target order
- for original `test` users, the last `prior` order is reassigned as the target order
- all orders earlier than the chosen target are the historical behavior used for feature construction

### 3. Basic cleaning

- fill `orders.days_since_prior_order` missing values with `0`
- sort orders within user by `order_number`
- remove literal `"missing"` metadata noise from the product hierarchy

The `"missing"` cleanup follows the earlier project notebooks:

- remove aisles whose name is literally `"missing"`
- remove departments whose name is literally `"missing"`
- remove products whose own name is `"missing"` or whose linked aisle/department is `"missing"`
- remove the associated rows from `order_products__prior` and `order_products__train`

### 4. Cumulative user timeline

Build a cumulative day index:

- `user_cum_days = cumulative sum of days_since_prior_order within user`

This enables later recency features in actual time, not only order count.

This cumulative timeline is exported as supporting context, but the main final feature set still follows the report definition based on:

- order-count recency
- average days between orders

### 5. Prior-detail interaction tables

Join prior order-product rows to order metadata so each prior interaction includes:

- `user_id`
- `order_number`
- `days_since_prior_order`
- `order_dow`
- `order_hour_of_day`
- `user_cum_days`

Then enrich with:

- `aisle_id`
- `department_id`
- `aisle`
- `department`

Two prior-detail views are maintained:

- `prior_detail`
  All prior interactions for the supervised users
- `history_detail`
  Only the prior interactions that occur strictly before the chosen target order for each user

The second table is the one used for candidate generation and later fold-wise feature construction.

### 6. Target label table

Create a deduplicated label table from `order_products__train` joined to user target orders:

- key: `(user_id, product_id)`
- value: `label = 1`

For original `test` users, labels come from the reassigned target order, which is the final prior order.

### 7. Candidate universe

Create the reorder candidate universe using only products already seen in the user prior history:

- key: `(user_id, product_id)`

This avoids open-catalog scoring and matches the reorder setting.

More precisely, candidates are built from `history_detail`, not from all prior interactions. This is important for original `test` users because their final prior order has been reassigned as target and must not remain in history.

### 8. Last-prior basket reference

For each user, identify:

- the last history order
- the set of products in that order
- the set of aisles in that order

These are reused later inside fold-wise Apriori and recent-basket logic.

## Fold-Wise Feature Engineering Rules

Within each training fold, the notebook builds features as follows.

### A. Base interaction features

Computed from training-fold prior history only:

- `up_buy_cnt`
- `up_reorder_cnt`
- `up_reorder_ratio`
- `up_first_order`
- `up_last_order`
- `up_avg_cart_order`
- `up_last_cum_days`
- `up_orders_since_last`
- `up_days_since_last`
- `up_freq`

### B. Product features

Computed from training-fold prior history only:

- `p_total_purchases`
- `p_unique_users`
- `p_reorder_ratio`
- `p_avg_cart_order`

### C. User features

Computed from training-fold prior history only:

- `u_total_items`
- `u_unique_products`
- `u_total_orders`
- `u_reorder_ratio`
- `u_avg_days_between_orders`
- `u_avg_basket_size`
- `u_last_cum_days`

### D. KMeans features

Workflow:

1. Build user-level clustering input features from training users only.
2. Apply log transforms where appropriate.
3. Fit `StandardScaler` on train users only.
4. Fit `KMeans(n_clusters=4)` on scaled train users only.
5. Assign train and evaluation users via train-fitted objects.
6. Create:
   - `user_cluster`
   - `user_cluster_1`
   - `user_cluster_2`
   - `user_cluster_3`

### E. Cluster-product target rate

This is the most leakage-sensitive feature.

It must be built only from training-fold labels:

- group by `(user_cluster, product_id)` within training candidates
- compute mean `label`
- merge the result into train and evaluation candidates

Fallback order for unseen combinations:

1. training `product_target_rate`
2. training `cluster_target_rate`
3. global training positive rate

### F. Apriori rule hit feature

Workflow:

1. Use training-fold prior baskets only.
2. Build aisle-level support, confidence, and lift statistics.
3. Filter by:
   - `support >= 0.01`
   - `confidence >= 0.30`
   - `lift >= 1.30`
4. Use the train-derived rule map to create `apriori_rule_hits` for both train and evaluation candidates, based on each user’s last prior basket aisles.

### G. ALS features

Workflow:

1. Fit ALS only on training-fold user-item counts.
2. Extract train-fitted user and item latent factors.
3. Compute:
   - `als_dot`
   - `als_cos`
4. For unseen users or products in evaluation rows, fall back to `0`.

## Evaluation Design

The notebook uses grouped nested cross-validation:

- outer CV: `GroupKFold`, grouped by `user_id`
- inner CV: `GroupKFold`, grouped by `user_id`
- tuning objective: `AUC`
- model comparison metrics:
  - `AUC`
  - `PR-AUC`
  - `LogLoss`
  - `Precision@11`
  - `Recall@11`
  - `F1@11`
  - `Hit@11`
  - dynamic-k analogs

## Downstream Analyses

After collecting outer-fold out-of-fold predictions, the notebook performs:

- final model comparison
- ablation study
- segment analysis by `user_cluster`
- top-k strategy comparison

All of these use leakage-safe out-of-fold predictions.
