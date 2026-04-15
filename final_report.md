# Instacart Grocery Reorder Prediction with User Segmentation, Apriori Rules, and ALS Features

## Team
Team 404 not Found

## 1. Executive Summary
This project studies the **next-order product prediction** problem on the Instacart Online Grocery Dataset. The practical goal is to support:

- product recommendation
- basket completion
- cross-sell and bundling
- personalized recommendation strategies based on user segments

We reformulated the raw Kaggle-style dataset into a supervised learning problem by treating the **last order with product-level labels for each user as the prediction target** and all earlier orders as historical behavior. On top of this, we combined:

- user-product interaction features
- user-level behavioral features
- product-level popularity and reorder features
- K-Means user segmentation
- Apriori aisle-level association features
- optional ALS collaborative filtering features

We evaluated four models:

- Logistic Regression
- LightGBM
- MLP
- CatBoost

The main findings are:

1. Nonlinear models clearly outperform Logistic Regression.
2. **CatBoost and MLP are the strongest models**, with CatBoost slightly ahead on `AUC` and MLP slightly ahead on `PR-AUC` and `F1@dynamic_k`.
3. Adding ALS features improves all models, but the gains are **small and consistent**, not transformative.
4. The strongest predictive signals still come from **user-product interaction and recency features**, while ALS acts as a modest complementary signal.

## 2. Business Motivation
The recommendation problem matters because online grocery shopping is highly repetitive but still contains discovery and bundling opportunities.

This project can support:

- **Conversion improvement** by surfacing likely next-order items
- **Basket size growth** through aisle-level co-purchase relationships
- **Personalization** by separating routine replenishment users from exploratory users
- **Operational targeting** by adapting recommendation strategy across user types

## 3. Dataset
We use the **Instacart Online Grocery Dataset** downloaded from KaggleHub.

The raw data includes:

- `orders`
- `order_products__prior`
- `order_products__train`
- `products`
- `aisles`
- `departments`

Original scale:

- about **3.4M orders**
- about **33.7M product-level purchase rows**
- about **206K users**

## 4. Data Preprocessing
The preprocessing pipeline was designed to make the dataset usable for supervised next-order prediction.

### 4.1 Cleaning
- Invalid `"missing"` categories were removed from the product hierarchy.
- `days_since_prior_order` missing values were treated as structural missingness from first orders and filled with `0`.

### 4.2 Task Redefinition
The original Kaggle split is not directly suitable for standard supervised learning because the test users do not provide product labels for their final order. To solve this:

- for each user, the **last labeled order** is treated as the target order
- all previous orders are treated as history

This produces a consistent user-level supervised learning setup without discarding large amounts of historical behavior.

### 4.3 Modeling Table Construction
The final modeling table is built at the `(user_id, product_id)` level:

- `label = 1` if the product appears in the user’s target order
- `label = 0` otherwise

Only historically purchased products are used as the candidate set, which keeps the recommendation problem realistic and computationally manageable.

## 5. Exploratory Data Analysis
EDA showed several important patterns that guided feature engineering.

### 5.1 User Heterogeneity
- Total orders per user are strongly long-tailed.
- Candidate size, defined as the number of unique products historically bought by a user, is also highly heterogeneous.

This supports user segmentation and user-specific recommendation length.

### 5.2 Basket Behavior
- Target basket sizes are right-skewed, with a typical order size in the mid single digits to low teens.
- User-product purchase counts are extremely long-tailed.

This confirms that **habitual repeat purchases** are central to the problem.

### 5.3 Recency Effects
- Reorder probability decreases monotonically with time/orders since last purchase.

This makes recency one of the most important modeling dimensions.

### 5.4 Temporal Patterns
- Orders cluster during daytime hours.
- Inter-order intervals show peaks around weekly cycles and at the dataset cap.

These patterns justify timing and cadence-based user features.

### 5.5 Product and Category Structure
- Product popularity is “head + long tail”.
- Reorder rates vary substantially across products, departments, and aisles.

This supports adding product-level and category-level repeatability features.

## 6. User Segmentation
We used **K-Means clustering** to capture heterogeneous shopping styles.

### 6.1 Clustering Features
Five interpretable user-level features were used:

- total historical orders
- average days between orders
- average basket size
- reorder rate
- number of unique products purchased

### 6.2 Why K-Means and Why K = 4
- PCA showed a meaningful low-dimensional structure in user behavior.
- K-Means was preferred over DBSCAN because DBSCAN produced highly imbalanced clusters and a large fraction of noise points.
- `K = 4` was chosen as a balance between statistical compactness and business interpretability.

### 6.3 Interpretable Segments
The four clusters broadly correspond to:

- light users with low commitment
- replenishment-oriented routine shoppers
- high-activity, high-loyalty core users
- broader-basket stock-up users

These segment labels were incorporated into modeling through dummy variables and the feature `cluster_product_target_rate`.

## 7. Basket-Level Association Analysis
We applied **Apriori** at the **aisle level** rather than the product level.

### 7.1 Why Aisle-Level Rules
- Product-level combinations are too sparse.
- Aisle-level rules are more stable and easier to interpret for bundling and cross-sell.

### 7.2 Rule Metrics
We used:

- support
- confidence
- lift

### 7.3 Final Thresholds
Based on sensitivity analysis and quality-coverage tradeoff:

- `support >= 0.01`
- `confidence >= 0.30`
- `lift >= 1.30`

These thresholds produced a moderate number of aisle-level rules with reasonable strength and coverage.

### 7.4 Modeling Use
The Apriori output was converted into the feature:

- `apriori_rule_hits`

This indicates whether a candidate product’s aisle is supported by association rules from the user’s recent basket context.

## 8. Feature Engineering
The final model-ready dataset included the following groups of features.

### 8.1 User-Product Interaction Features
- `up_orders_since_last`
- `up_days_since_last`
- `up_freq`
- `up_buy_cnt`
- `up_reorder_ratio`
- `up_first_order`
- `up_last_order`
- `up_avg_cart_order`

These are the most important features in the project.

### 8.2 Product Features
- `p_reorder_ratio`
- `p_total_purchases`
- `p_unique_users`
- `p_avg_cart_order`

### 8.3 User Features
- `u_total_orders`
- `u_reorder_ratio`
- `u_unique_products`
- `u_total_items`
- `u_avg_days_between_orders`
- `u_avg_basket_size`

### 8.4 Segmentation Features
- `user_cluster_1`
- `user_cluster_2`
- `user_cluster_3`
- `cluster_product_target_rate`

### 8.5 Association Feature
- `apriori_rule_hits`

### 8.6 ALS Features
- `als_dot`
- `als_cos`

These were treated as optional auxiliary features and used only for the “with ALS” experiments.

### 8.7 Perfect Correlation Cleanup
During feature analysis, `up_buy_cnt` and `up_reorder_cnt` were found to be perfectly correlated in the constructed table. We therefore:

- **kept `up_buy_cnt`**
- **dropped `up_reorder_cnt`**

The final dataset used in the report does **not** include `up_reorder_cnt`.

## 9. Modeling Framework
We formulate the task as binary classification over candidate user-product pairs, then convert predicted probabilities into ranked recommendation lists.

### 9.1 Models
- **Logistic Regression (SAGA)** as the linear baseline
- **LightGBM** as a strong tree-based tabular baseline
- **MLP** as a nonlinear neural baseline
- **CatBoost** as a boosted tree model with explicit categorical support

### 9.2 Recommendation Strategy
Two recommendation outputs are evaluated:

- fixed-length `Top-11`
- `dynamic-k`, where predicted list length is based on user historical average basket size

### 9.3 Evaluation Metrics
Row-level metrics:

- `AUC`
- `PR-AUC`
- `LogLoss`

Recommendation metrics:

- `Precision@11`
- `Recall@11`
- `F1@11`
- `Hit@11`
- `Precision@dynamic_k`
- `Recall@dynamic_k`
- `F1@dynamic_k`
- `Hit@dynamic_k`

## 10. Experimental Design
The final notebook used a single full prepared dataset and defined two feature views internally:

- **no_als**: base features only
- **with_als**: base features plus `als_dot` and `als_cos`

Key design choices:

- CatBoost was tuned **only on the no-ALS feature set**, to stay consistent with the project’s earlier no-ALS tuning logic.
- All four models were then evaluated on:
  - no ALS
  - with ALS
- Grouped cross-validation by user was used to avoid leakage across the same user.

## 11. Final Results

### 11.1 Mean Cross-Validated Performance

| Feature Set | Model | AUC | PR-AUC | LogLoss | F1@11 | F1@dynamic_k |
|---|---|---:|---:|---:|---:|---:|
| no_als | CatBoost | 0.846121 | 0.434539 | 0.485951 | 0.352327 | 0.383553 |
| with_als | CatBoost | 0.846831 | 0.434831 | 0.485259 | 0.352733 | 0.383707 |
| no_als | LightGBM | 0.845192 | 0.434176 | 0.239876 | 0.351724 | 0.382606 |
| with_als | LightGBM | 0.845923 | 0.434648 | 0.239565 | 0.352213 | 0.383118 |
| no_als | LogisticRegression_SAGA | 0.833910 | 0.406281 | 0.504988 | 0.342465 | 0.369833 |
| with_als | LogisticRegression_SAGA | 0.834443 | 0.406417 | 0.504515 | 0.342667 | 0.370027 |
| no_als | MLP | 0.845483 | 0.435824 | 0.239695 | 0.352378 | 0.383544 |
| with_als | MLP | 0.846185 | 0.436255 | 0.239461 | 0.352615 | 0.383757 |

### 11.2 Model Ranking
Using `PR-AUC` and `F1@dynamic_k` as the most relevant selection criteria:

- **Best PR-AUC**: `MLP + ALS` (`0.436255`)
- **Best F1@dynamic_k**: `MLP + ALS` (`0.383757`)
- **Best AUC**: `CatBoost + ALS` (`0.846831`)

Overall, **MLP + ALS** is the most balanced top model in this project, while **CatBoost + ALS** is a very close competitor.

### 11.3 ALS Improvement

| Model | Δ AUC | Δ PR-AUC | Δ F1@11 | Δ F1@dynamic_k | Δ LogLoss |
|---|---:|---:|---:|---:|---:|
| CatBoost | +0.000710 | +0.000293 | +0.000406 | +0.000155 | -0.000693 |
| LightGBM | +0.000730 | +0.000472 | +0.000489 | +0.000513 | -0.000311 |
| LogisticRegression_SAGA | +0.000534 | +0.000136 | +0.000202 | +0.000194 | -0.000473 |
| MLP | +0.000702 | +0.000432 | +0.000237 | +0.000213 | -0.000234 |

Interpretation:

- ALS improves every model on average.
- The gain is **real but modest**.
- The largest `F1@dynamic_k` improvement appears in **LightGBM**.
- The largest `PR-AUC` and strongest overall final performance appear in **MLP + ALS**.

## 12. Discussion

### 12.1 Why Logistic Regression Underperforms
Logistic Regression is useful as a baseline, but the recommendation problem contains nonlinear relationships among:

- recency
- interaction frequency
- user heterogeneity
- cluster-level item propensity
- association-based basket signals

This makes linear decision boundaries too limited compared with tree-based and neural models.

### 12.2 Why CatBoost Did Not Dramatically Dominate
Literature often reports strong CatBoost performance on tabular problems, but our task is heavily driven by **high-quality numerical interaction features**, not only raw categorical variables. In this setting:

- LightGBM already performs strongly
- MLP can exploit the engineered nonlinear structure
- CatBoost remains competitive, but does not dominate by a wide margin

This is a reasonable and informative experimental outcome.

### 12.3 Why ALS Only Helps a Little
ALS contributes a collaborative filtering signal, but the core problem is already captured well by:

- user-product purchase history
- recency features
- reorder statistics
- user cluster interaction features

So ALS acts as a **complementary refinement**, not a replacement for the hand-crafted interaction features.

## 13. Main Conclusions
The project supports the following conclusions:

1. The Instacart next-order prediction problem can be effectively modeled as candidate-level supervised learning.
2. User-product interaction and recency features are the strongest predictors.
3. User segmentation and Apriori basket rules provide useful contextual information.
4. ALS features improve results consistently, but only by a small margin.
5. The best overall final model in this study is **MLP with ALS**, with CatBoost and LightGBM remaining very close alternatives.

## 14. Limitations
- Candidate generation only uses historically purchased products, so pure cold-start discovery is not addressed.
- ALS gains are small, suggesting that richer collaborative signals or larger embedding pipelines may be needed for larger improvements.
- We did not perform a formal statistical significance test across folds.
- CatBoost tuning was intentionally constrained to remain aligned with the earlier no-ALS tuning pipeline.

## 15. Future Work
- test richer candidate expansion strategies beyond historical products
- perform feature ablation for segmentation, Apriori, and ALS separately
- investigate calibration and thresholding for probability-to-list conversion
- try deeper ranking-oriented objectives instead of pure binary classification
- analyze performance by user segment to understand which user groups benefit most from ALS and dynamic-k

## 16. Output Files
The final experiment outputs are stored in:

- [final_model_full_run_summary.csv](/Users/bettinabopeng/Documents/GitHub/5971/artifacts/final_model_full_run_summary.csv)
- [final_model_als_delta_summary.csv](/Users/bettinabopeng/Documents/GitHub/5971/artifacts/final_model_als_delta_summary.csv)
- [catboost_tuning_no_als.csv](/Users/bettinabopeng/Documents/GitHub/5971/artifacts/catboost_tuning_no_als.csv)
- [model_dataset_full.parquet](/Users/bettinabopeng/Documents/GitHub/5971/artifacts/model_dataset_full.parquet)
