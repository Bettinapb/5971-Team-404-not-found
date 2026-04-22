# Instacart Grocery Reorder Prediction with User Segmentation, Apriori Rules, and ALS Features

## Team
Team 404 not Found

## 1. Executive Summary
This document is the comprehensive final write-up of our Instacart project. The goal of the project is to predict which products a user is most likely to buy in their next grocery order. We formulate this as a candidate-level supervised learning problem, build a large feature table from historical orders, and evaluate multiple ranking-oriented recommendation strategies.

The full project evolved across several stages:

- [midterm_team_404_not_found.ipynb](/Users/bettinabopeng/Documents/GitHub/5971/midterm_team_404_not_found.ipynb): business framing, data cleaning, EDA, initial KMeans segmentation, Apriori association analysis, and the overall recommendation concept
- [3over4_team_404_not_found.ipynb](/Users/bettinabopeng/Documents/GitHub/5971/3over4_team_404_not_found.ipynb): stronger feature-engineering justification, initial holdout modeling, baseline comparison, dynamic-k evaluation, and optional ALS exploration
- [new.ipynb](/Users/bettinabopeng/Documents/GitHub/5971/new.ipynb): feature-aligned nested CV tuning with KMeans and Apriori features, plus more disciplined Top-K evaluation
- [build_model_dataset.py](/Users/bettinabopeng/Documents/GitHub/5971/build_model_dataset.py): consolidated end-to-end preprocessing and export of the final modeling table
- [final_model_full_run.ipynb](/Users/bettinabopeng/Documents/GitHub/5971/final_model_full_run.ipynb): frozen final comparison of Logistic Regression, LightGBM, MLP, and CatBoost, each with and without ALS
- [supplementary_model_analysis.ipynb](/Users/bettinabopeng/Documents/GitHub/5971/supplementary_model_analysis.ipynb): ablation study, segment-level evaluation, and Top-K strategy comparison

The final conclusions are:

1. The next-order prediction problem can be modeled effectively as supervised learning on `(user, product)` candidates.
2. The strongest predictive signals come from user-product interaction history and recency.
3. KMeans-derived features create the largest jump beyond the basic historical feature set.
4. Apriori and ALS both help, but their gains are relatively small.
5. Among the final models, MLP with ALS is the strongest overall model on the most relevant recommendation metrics, while CatBoost with ALS achieves the highest AUC.
6. A personalized recommendation length based on historical basket size performs better than a fixed `k = 11`.

## 2. Business Problem and Motivation
Instacart is a grocery shopping platform, so recommendation quality depends on both repeated purchase behavior and basket context.

This task matters because a strong next-order prediction system can support:

- product recommendation
- basket completion
- faster reorder workflows
- cross-sell and bundling
- user-segment-specific recommendation strategies

Grocery recommendation is different from many other recommendation tasks because it combines:

- habitual repeat behavior, such as buying the same milk, eggs, or bread regularly
- contextual co-purchase behavior, such as buying salsa and tortilla chips together
- user heterogeneity, where some users are heavy routine shoppers while others are more exploratory

Our project aims to model all three.

## 3. Dataset
We use the Instacart Online Grocery Dataset. The raw data contain approximately:

- 3.4 million orders
- 33.7 million product-level order lines
- 206 thousand users

The core raw tables are:

- `orders.csv`
- `order_products__prior.csv`
- `order_products__train.csv`
- `products.csv`
- `aisles.csv`
- `departments.csv`

In the final exported modeling dataset, after user-level task reformulation and candidate construction, we have:

- 8,474,661 candidate rows
- 131,209 users
- positive rate about `9.78%`

These values are recorded in [model_dataset_full.metadata.json](/Users/bettinabopeng/Documents/GitHub/5971/artifacts/model_dataset_full.metadata.json).

## 4. Problem Reformulation
The raw Kaggle setup is not directly suitable for standard supervised learning because the test users do not provide product-level labels for their final order.

To solve this, we reformulate the task as follows:

1. For each user, treat the last order with product-level information as the target order.
2. Treat all earlier orders as historical behavior.
3. Construct candidate rows at the `(user, product)` level.
4. Assign `label = 1` if the candidate product appears in the target order, and `label = 0` otherwise.

This turns the problem into a next-order prediction task that is directly compatible with supervised learning and ranking evaluation.

## 5. Data Cleaning and Preprocessing
The final preprocessing pipeline is implemented in [build_model_dataset.py](/Users/bettinabopeng/Documents/GitHub/5971/build_model_dataset.py).

The main preprocessing steps are:

1. Load raw Instacart tables either from a local directory or automatically through `kagglehub`.
2. Keep `orders` from `eval_set = prior` as history and `eval_set = train` as the user target order.
3. Remove invalid `"missing"` category noise in the product hierarchy where needed.
4. Fill `days_since_prior_order` missing values with `0` for first orders.
5. Join product taxonomy information such as `aisle_id` and `department_id`.
6. Build target-order labels and history-derived user, product, and user-product aggregates.
7. Add KMeans, Apriori, and optionally ALS features.
8. Export a final parquet modeling table and metadata JSON.

## 6. Candidate Generation
The project uses a practical and conservative candidate-generation strategy:

- for each user, only previously purchased products are included as candidates

This has two important consequences:

1. It keeps the candidate space tractable.
2. It makes the positive class imbalance much less extreme than recommending over the full catalog.

This strategy does not solve cold-start recommendation for completely unseen products, but it is a strong fit for grocery reorder prediction, where repeated purchase behavior is very common.

## 7. Exploratory Data Analysis and Early Insights
The earliest notebooks established several core facts that remained important throughout the project.

### 7.1 User Heterogeneity
Users differ substantially in:

- total number of orders
- number of unique purchased products
- reorder tendency
- basket size
- time between orders

This strongly motivated user segmentation.

### 7.2 Long-Tail Interaction Structure
Basket sizes and user-product purchase counts are highly skewed:

- a small number of products and user-product pairs are purchased repeatedly
- most user-product relationships are sparse

This supports feature engineering around both frequent-repeat and sparse-repeat behavior.

### 7.3 Strong Recency Effect
One of the strongest EDA findings is the monotonic relationship between recency and reorder probability:

- recently purchased products are much more likely to reappear
- products not bought for many orders are much less likely to be reordered

This is why features like `up_orders_since_last` and `up_days_since_last` are central in the final models.

### 7.4 Product, Department, and Aisle Structure
The product hierarchy is useful because:

- some products have intrinsically high reorder rates
- some aisles support stable co-purchase patterns
- category-level context is useful for cross-sell and segment-aware recommendation

## 8. User Segmentation with KMeans
User segmentation was introduced in the early phase and then integrated directly into the final modeling pipeline.

### 8.1 Clustering Features
We cluster users using five user-level summary features:

- `u_total_orders`
- `u_avg_days_between_orders`
- `u_avg_basket_size`
- `u_reorder_ratio`
- `u_unique_products`

These collectively describe:

- activity level
- order rhythm
- basket scale
- loyalty
- exploration breadth

### 8.2 Why KMeans
KMeans was chosen because:

- it is computationally efficient on large tabular datasets
- it is easy to explain to a business audience
- the feature space is low-dimensional and well suited to centroid-based clustering

PCA analysis showed that the first two principal components explain about `76.5%` of the variance, which supports a relatively low-dimensional behavioral representation.

### 8.3 Why `K = 4`
We tested `K = 2` to `K = 9`.

Observed behavior:

- inertia decreases steadily without a strong elbow
- silhouette is highest at small `K`
- the overall feature space looks more continuous than naturally separated

Therefore, `K = 4` was selected as a compromise between:

- statistical compactness
- business interpretability

### 8.4 Why Not DBSCAN
DBSCAN was also considered, but in our data:

- most users collapsed into one large cluster
- small clusters were unstable
- about `23%` of points were labeled as noise

This made DBSCAN less suitable for interpretable downstream modeling.

### 8.5 How Segmentation Enters the Model
Segmentation enters the final supervised models through:

- `user_cluster_1`
- `user_cluster_2`
- `user_cluster_3`
- `cluster_product_target_rate`

The raw `user_cluster` label remains in the exported dataset for analysis, but the main models use the one-hot encoding and the cluster-product interaction rate.

## 9. Basket-Level Association Analysis with Apriori
We use Apriori to capture basket context.

### 9.1 Why Aisle-Level Rules
We chose aisle-level rather than product-level rules because:

- product-level combinations are too sparse
- aisle-level patterns are more stable
- aisle-level rules are easier to interpret for bundling and cross-sell

### 9.2 Rule Definitions
For a rule `A -> B`:

- `support(A, B) = #(A, B) / #(all baskets)`
- `confidence(A -> B) = support(A, B) / support(A) = P(B | A)`
- `lift(A -> B) = P(A, B) / (P(A) * P(B))`

Interpretation:

- support measures how frequent a pair is overall
- confidence measures how likely `B` is when `A` appears
- lift measures whether the co-occurrence is stronger than chance

### 9.3 Threshold Selection
Thresholds were chosen through sensitivity and tradeoff analysis rather than arbitrarily.

The final settings are:

- `support = 0.01`
- `confidence = 0.30`
- `lift = 1.30`

These thresholds balance:

- rule quality
- rule coverage
- interpretability

The final rule set contains `138` filtered aisle-level rules.

### 9.4 How Apriori Enters the Model
Apriori is not used as a standalone recommender. Instead, it becomes one feature:

- `apriori_rule_hits`

Construction logic:

1. Take aisles from the user’s most recent historical basket.
2. Check whether the candidate product’s aisle is a rule consequent of any recent-basket aisle.
3. Mark a positive hit if so.

This lets the supervised model absorb basket-context signals without replacing the ranking model.

## 10. ALS and Collaborative Filtering Features
ALS was explored first in [code.ipynb](/Users/bettinabopeng/Documents/GitHub/5971/code.ipynb), formalized in later notebooks, and consolidated in [build_model_dataset.py](/Users/bettinabopeng/Documents/GitHub/5971/build_model_dataset.py).

### 10.1 ALS Principle
ALS stands for Alternating Least Squares. It is a matrix factorization method for recommendation.

The user-item interaction matrix is approximated as:

`R_ij ~= u_i^T v_j`

where:

- `R_ij` is the observed interaction strength between user `i` and item `j`
- `u_i` is the latent factor vector for user `i`
- `v_j` is the latent factor vector for product `j`

ALS alternates between:

- solving user factors while holding item factors fixed
- solving item factors while holding user factors fixed

In the final pipeline we use:

- `factors = 32`
- `regularization = 0.01`
- `iterations = 15`
- confidence scaling `alpha = 15`

### 10.2 ALS Features Used
We do not use ALS as the final recommender. We use ALS as a feature generator:

- `als_dot = u_i^T v_j`
- `als_cos = (u_i^T v_j) / (||u_i|| * ||v_j||)`

Interpretation:

- `als_dot` measures raw latent affinity
- `als_cos` measures normalized directional similarity

These features supplement the supervised models with collaborative similarity signals.

## 11. Final Feature Engineering
The final exported dataset contains 31 columns total, including identifiers, label, analysis columns, and modeling features.

### 11.1 Final Base Feature Set
The final base modeling feature list is:

- `up_orders_since_last`
- `up_days_since_last`
- `up_freq`
- `up_buy_cnt`
- `up_reorder_ratio`
- `p_reorder_ratio`
- `u_total_orders`
- `cluster_product_target_rate`
- `p_avg_cart_order`
- `u_reorder_ratio`
- `u_unique_products`
- `p_total_purchases`
- `up_first_order`
- `u_avg_days_between_orders`
- `p_unique_users`
- `up_last_order`
- `u_total_items`
- `u_avg_basket_size`
- `up_avg_cart_order`
- `apriori_rule_hits`
- `user_cluster_1`
- `user_cluster_2`
- `user_cluster_3`

Optional ALS features:

- `als_dot`
- `als_cos`

Additional columns needed by CatBoost:

- `aisle_id`
- `department_id`

### 11.2 Feature Group Definitions

#### User-product interaction features
These are the strongest features in the project.

- `up_orders_since_last`: how many orders have passed since the user last bought the product
- `up_days_since_last`: how many days have passed since the user last bought the product
- `up_freq`: purchase frequency of the user-product pair
- `up_buy_cnt`: total purchase count of the user-product pair
- `up_reorder_ratio`: repeat-purchase ratio of the user-product pair
- `up_first_order`: first order number in which the user bought the product
- `up_last_order`: last historical order number in which the user bought the product
- `up_avg_cart_order`: average add-to-cart position for the user-product pair

These features capture habit strength and recency directly.

#### Product-level features
- `p_reorder_ratio`: overall reorder tendency of the product
- `p_avg_cart_order`: average position of the product within baskets
- `p_total_purchases`: total times the product appears
- `p_unique_users`: number of distinct users who bought the product

These capture product popularity, repeatability, and broad usage.

#### User-level features
- `u_total_orders`: total historical orders for the user
- `u_reorder_ratio`: overall reorder tendency of the user
- `u_unique_products`: number of unique products bought by the user
- `u_avg_days_between_orders`: average time gap between orders
- `u_total_items`: total purchased items in history
- `u_avg_basket_size`: average number of items per order

These capture activity level and shopping style.

#### Segmentation features
- `user_cluster_1`
- `user_cluster_2`
- `user_cluster_3`
- `cluster_product_target_rate`

These let the model combine personal history with cluster-level priors.

#### Apriori feature
- `apriori_rule_hits`

This captures aisle-level co-purchase support from the user’s most recent basket context.

#### ALS features
- `als_dot`
- `als_cos`

These capture latent collaborative similarity.

### 11.3 Perfect-Correlation Cleanup
Feature analysis in [3over4_team_404_not_found.ipynb](/Users/bettinabopeng/Documents/GitHub/5971/3over4_team_404_not_found.ipynb) revealed that:

- `up_buy_cnt`
- `up_reorder_cnt`

were perfectly correlated in the final table.

Therefore:

- `up_buy_cnt` was kept
- `up_reorder_cnt` was removed

This cleanup was enforced in the final pipeline and checked again in the final notebooks.

## 12. Modeling Approaches and Their Principles
We ultimately compare four models.

### 12.0 Hyperparameter Tuning Strategy
Because candidate rows from the same user are highly dependent, we did not use ordinary random train/validation splits for model selection. Instead, we tuned the final models with grouped cross-validation so that all rows from a user stay in the same fold.

#### 12.0.1 Grouped Nested CV Design
The tuning logic is implemented across:

- [new.ipynb](/Users/bettinabopeng/Documents/GitHub/5971/new.ipynb) for Logistic Regression, LightGBM, and MLP
- [final_model_full_run.ipynb](/Users/bettinabopeng/Documents/GitHub/5971/final_model_full_run.ipynb) for CatBoost and the frozen final comparison

The validation design is:

- grouping variable: `user_id`
- outer CV: `GroupKFold(n_splits=3)`
- inner CV: `GroupKFold(n_splits=3)`
- search objective inside tuning: `roc_auc`
- random seed: `42`

Interpretation:

- the outer fold estimates generalization performance
- the inner fold is used only for hyperparameter selection
- grouping by `user_id` prevents leakage across the same user’s candidate rows

This means that for any outer split:

- users in the test fold are completely unseen during tuning
- users in the inner validation folds are also disjoint from the inner training folds

So the model is never tuned and evaluated on candidate rows from the same customer at the same time.

#### 12.0.2 Tuning Data Scope
We used two related but slightly different tuning stages.

For Logistic Regression, LightGBM, and MLP:

- tuning was first done in [new.ipynb](/Users/bettinabopeng/Documents/GitHub/5971/new.ipynb)
- to keep nested CV computationally manageable during iteration, that notebook uses `SAMPLE_USERS = 20000`
- the feature set was already aligned with the final engineered table structure
- after tuning, one frozen configuration per model was selected and reused in the full final run

For CatBoost:

- tuning was done later in [final_model_full_run.ipynb](/Users/bettinabopeng/Documents/GitHub/5971/final_model_full_run.ipynb)
- the full exported modeling table `artifacts/model_dataset_full.parquet` was used
- CatBoost was tuned only on the `no_als` feature view first
- the resulting fixed CatBoost configuration was then reused for both `no_als` and `with_als` evaluation

This separation was intentional:

- the earlier notebook identified strong stable settings for the first three models
- the final notebook then froze those settings and performed a clean apples-to-apples comparison on the full dataset

#### 12.0.3 Search Method
For all four models, the search method was `RandomizedSearchCV` rather than exhaustive grid search.

The exact setup was:

- Logistic Regression: `n_iter = 12`
- LightGBM: `n_iter = 12`
- MLP: `n_iter = 10`
- CatBoost: `n_iter = 12`

Other important search settings were:

- `scoring = "roc_auc"`
- `refit = True`
- `random_state = 42`

For Logistic Regression, LightGBM, and MLP, `n_jobs = -1` was used during search.

For CatBoost, `n_jobs = 1` was used in the search stage to avoid instability from combining CatBoost’s own training behavior with outer parallel search.

#### 12.0.4 Preprocessing Inside Tuning
The tuning pipelines matched the needs of each model.

For Logistic Regression:

- a `StandardScaler` was placed inside the pipeline
- the classifier was `LogisticRegression(solver="saga", class_weight="balanced")`

For MLP:

- a `StandardScaler` was also used inside the pipeline
- the base estimator used `early_stopping = True`
- `validation_fraction = 0.1`
- `n_iter_no_change = 8`
- `max_iter = 80`

For LightGBM:

- no feature scaling pipeline was used
- the base estimator was `LGBMClassifier(objective="binary")`

For CatBoost:

- no scaling was used
- categorical columns were passed explicitly as `aisle_id` and `department_id`
- the base estimator used `loss_function = "Logloss"` and `eval_metric = "AUC"`
- `allow_writing_files = False` was set so that tuning stayed self-contained

#### 12.0.5 Logistic Regression Tuning Details
The Logistic Regression search space in [new.ipynb](/Users/bettinabopeng/Documents/GitHub/5971/new.ipynb) was:

- `C in {0.03, 0.1, 0.3, 1.0, 3.0}`
- `max_iter in {150, 250, 400, 600}`

All three outer folds selected the same best configuration:

- `C = 0.03`
- `max_iter = 150`

So the final frozen Logistic Regression model used:

- `solver = "saga"`
- `class_weight = "balanced"`
- `C = 0.03`
- `max_iter = 150`

#### 12.0.6 LightGBM Tuning Details
The LightGBM search space in [new.ipynb](/Users/bettinabopeng/Documents/GitHub/5971/new.ipynb) was:

- `num_leaves in {31, 63, 127}`
- `learning_rate in {0.02, 0.03, 0.05, 0.1}`
- `n_estimators in {250, 400, 600}`
- `min_child_samples in {20, 30, 50, 100}`
- `subsample in {0.8, 1.0}`
- `colsample_bytree in {0.8, 1.0}`

The nested-CV winners were concentrated around a conservative boosting regime with:

- small-to-moderate tree size
- low learning rate
- relatively long boosting horizon
- partial row subsampling

The final frozen LightGBM configuration was:

- `num_leaves = 31`
- `learning_rate = 0.03`
- `n_estimators = 600`
- `min_child_samples = 30`
- `subsample = 0.8`
- `colsample_bytree = 1.0`

This setting reflects the dominant stable pattern from tuning rather than chasing a single fold-specific winner.

#### 12.0.7 MLP Tuning Details
The MLP search space in [new.ipynb](/Users/bettinabopeng/Documents/GitHub/5971/new.ipynb) was:

- `hidden_layer_sizes in {(64,), (128,), (128, 64)}`
- `alpha in {1e-4, 3e-4, 1e-3, 3e-3}`
- `learning_rate_init in {3e-4, 6e-4, 1e-3}`
- `batch_size in {256, 512, 1024}`

The search was deliberately conservative because:

- the candidate table is large
- the labels are imbalanced
- user-product negatives are noisy
- we wanted a stable neural baseline rather than an overfit network

The final frozen MLP configuration was:

- `hidden_layer_sizes = (128, 64)`
- `alpha = 1e-3`
- `learning_rate_init = 1e-3`
- `batch_size = 512`
- `early_stopping = True`
- `validation_fraction = 0.1`
- `n_iter_no_change = 8`
- `max_iter = 80`

#### 12.0.8 CatBoost Tuning Details
CatBoost was tuned separately in [final_model_full_run.ipynb](/Users/bettinabopeng/Documents/GitHub/5971/final_model_full_run.ipynb) on the full exported dataset.

The CatBoost search space was:

- `depth in {6, 8, 10}`
- `learning_rate in {0.03, 0.05, 0.08}`
- `l2_leaf_reg in {3.0, 5.0, 7.0, 9.0}`
- `iterations in {400, 700, 1000}`
- `subsample in {0.8, 1.0}`

Two additional details matter here.

First, CatBoost used the categorical columns:

- `aisle_id`
- `department_id`

Second, class imbalance was handled fold by fold through:

- `scale_pos_weight = (1 - pos_rate) / pos_rate`

where `pos_rate` was computed from the outer-training portion of each split.

The outer-fold tuning winners were very similar, centering on:

- `depth = 8`
- `learning_rate = 0.03`
- `l2_leaf_reg = 9.0`
- `iterations = 1000`
- `subsample = 0.8` in most folds

So the final frozen CatBoost configuration for the no-ALS and with-ALS evaluations was:

- `depth = 8`
- `iterations = 1000`
- `learning_rate = 0.03`
- `l2_leaf_reg = 9.0`
- `subsample = 0.8`

#### 12.0.9 Final Frozen Evaluation Protocol
After tuning, we did not re-tune separately inside every final feature-set comparison. Instead, we froze one configuration per model and evaluated those fixed estimators under the same grouped outer CV.

This final comparison uses:

- `GroupKFold(n_splits=3)` on the full modeling table
- the `no_als` feature set
- the `with_als` feature set
- the same order-level ranking metrics for every model

This protocol is important because it keeps the comparison fair:

- every model sees the same user-level splits
- every model is judged with the same frozen hyperparameters
- the difference between `no_als` and `with_als` reflects feature effects rather than fresh hyperparameter tuning noise

### 12.1 Logistic Regression
Logistic Regression is the linear baseline.

Form:

`P(y = 1 | x) = sigmoid(w^T x + b)`

with:

- `x` the feature vector
- `w` the coefficient vector
- `b` the bias term
- `sigmoid(z) = 1 / (1 + exp(-z))`

Strengths:

- simple
- interpretable
- stable baseline

Weakness:

- cannot naturally capture strong nonlinear interactions

Final fixed settings:

- solver `saga`
- `class_weight = balanced`
- `C = 0.03`
- `max_iter = 150`

### 12.2 LightGBM
LightGBM is a gradient boosting decision tree model.

Generic boosting form:

`F_M(x) = sum_{m=1 to M} gamma_m * h_m(x)`

where each `h_m(x)` is a tree trained to reduce the remaining errors.

Strengths:

- strong on tabular data
- captures nonlinear relationships
- captures feature interactions
- efficient training on large datasets

Final fixed settings:

- `num_leaves = 31`
- `learning_rate = 0.03`
- `n_estimators = 600`
- `min_child_samples = 30`
- `subsample = 0.8`
- `colsample_bytree = 1.0`

### 12.3 MLP
MLP is a multilayer perceptron with hidden layers.

For one hidden layer:

- `h = phi(W1 x + b1)`
- `y_hat = sigmoid(W2 h + b2)`

Strengths:

- learns nonlinear mappings
- effective when good engineered features already exist

Final fixed settings:

- `hidden_layer_sizes = (128, 64)`
- `alpha = 1e-3`
- `learning_rate_init = 1e-3`
- `batch_size = 512`
- `max_iter = 80`
- `early_stopping = True`

### 12.4 CatBoost
CatBoost is another gradient-boosted tree model.

Like LightGBM, it is an additive tree ensemble:

`F_M(x) = sum_{m=1 to M} gamma_m * h_m(x)`

Strengths:

- strong on mixed tabular data
- handles categorical structure well
- uses ordered boosting to reduce training bias

In our project, CatBoost was evaluated with explicit categorical columns:

- `aisle_id`
- `department_id`

Final tuned parameters on the no-ALS setup:

- `depth = 8`
- `iterations = 1000`
- `learning_rate = 0.03`
- `l2_leaf_reg = 9.0`
- `subsample = 0.8` in the majority of folds

## 13. Evaluation Metrics and Their Meaning
We evaluate both row-level probability quality and order-level recommendation quality.

### 13.1 Row-level metrics

#### AUC
AUC is the area under the ROC curve.

- `TPR = TP / (TP + FN)`
- `FPR = FP / (FP + TN)`

It measures how well the model ranks positives above negatives across all thresholds.

#### PR-AUC
PR-AUC is the area under the precision-recall curve.

- `Precision = TP / (TP + FP)`
- `Recall = TP / (TP + FN)`

Because the positive rate is only about `9.78%`, PR-AUC is especially important in this project.

#### LogLoss
`LogLoss = -(1 / N) * sum_i [ y_i * log(p_hat_i) + (1 - y_i) * log(1 - p_hat_i) ]`

This measures the quality of predicted probabilities. Lower is better.

### 13.2 Order-level recommendation metrics

#### Precision@k
Among the `k` recommended products, how many are correct on average.

#### Recall@k
Among the true purchased products, how many are recovered in the top `k`.

#### F1@k
`F1 = 2 * Precision * Recall / (Precision + Recall)`

This balances precision and recall.

#### Hit@k
`Hit@k = 1` if at least one true product appears in the top `k`, otherwise `0`.

This measures whether the recommender gets at least one product right for a user.

#### F1@11
This uses a fixed recommendation length of 11 items per user.

#### F1@dynamic_k
This uses a personalized `k`, based on the rounded historical average basket size of each user.

This is more realistic for grocery shopping because not all users buy the same number of products.

## 14. Experimental Design
The project used multiple evaluation stages.

### 14.1 Early holdout stage
In the 3/4 notebook, we used:

- user-level train/validation/test split
- fixed best parameters from earlier tuning
- baseline Top-11 and dynamic-k comparison

### 14.2 Nested cross-validation stage
In [new.ipynb](/Users/bettinabopeng/Documents/GitHub/5971/new.ipynb), we used:

- user-grouped nested CV
- aligned feature set
- tuning for Logistic Regression, LightGBM, and MLP

This stage showed that:

- LightGBM and MLP clearly outperform Logistic Regression
- dynamic-k is better than fixed Top-11

### 14.3 Final frozen comparison
In [final_model_full_run.ipynb](/Users/bettinabopeng/Documents/GitHub/5971/final_model_full_run.ipynb), we froze the pipeline and compared:

- Logistic Regression
- LightGBM
- MLP
- CatBoost

under two feature settings:

- `no_als`
- `with_als`

All results use `GroupKFold(n_splits = 3)` by user.

## 15. Main Final Results
The final cross-validated results are:

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

These results come from [final_model_full_run_summary.csv](/Users/bettinabopeng/Documents/GitHub/5971/artifacts/final_model_full_run_summary.csv).

### 15.1 Interpretation
There are three major takeaways:

1. All nonlinear models outperform Logistic Regression.
2. MLP, CatBoost, and LightGBM are very close, but MLP with ALS is the strongest overall by recommendation-oriented metrics.
3. CatBoost with ALS has the highest AUC.

### 15.2 Why LogLoss Looks Different Across Models
The absolute scale of LogLoss is much lower for LightGBM and MLP than for Logistic Regression and CatBoost in the saved summaries. This suggests differences in probability calibration and output sharpness, not just ranking quality. Since this project ultimately optimizes recommendation ranking quality, PR-AUC and `F1@dynamic_k` are the more important metrics for model selection.

## 16. ALS vs. No ALS
The average incremental effect of ALS is:

| Model | Δ AUC | Δ PR-AUC | Δ F1@11 | Δ F1@dynamic_k | Δ LogLoss |
|---|---:|---:|---:|---:|---:|
| CatBoost | +0.000710 | +0.000293 | +0.000406 | +0.000155 | -0.000693 |
| LightGBM | +0.000730 | +0.000472 | +0.000489 | +0.000513 | -0.000311 |
| LogisticRegression_SAGA | +0.000534 | +0.000136 | +0.000202 | +0.000194 | -0.000473 |
| MLP | +0.000702 | +0.000432 | +0.000237 | +0.000213 | -0.000234 |

Interpretation:

- ALS improves every model on average
- the gains are consistent but small
- ALS is a complementary feature, not the primary driver of performance

The largest `F1@dynamic_k` improvement appears in LightGBM.

## 17. Feature-Group Ablation Study
The supplementary ablation notebook explicitly measured the contribution of each feature group. Stages were:

1. interaction only
2. + user and product features
3. + KMeans features
4. + Apriori feature
5. + ALS features

### 17.1 Main Ablation Findings
Across all models, the pattern is consistent:

- adding user and product features gives a clear improvement over interaction-only features
- adding KMeans features gives the largest jump beyond that
- adding Apriori gives only a very small incremental gain
- adding ALS gives another small but consistent gain

Example with LightGBM on `F1@dynamic_k`:

- `interaction_only`: `0.358680`
- `plus_user_product`: `0.368873`
- `plus_kmeans`: `0.382522`
- `plus_apriori`: `0.382606`
- `plus_als`: `0.383118`

Example with MLP:

- `0.358576 -> 0.368849 -> 0.383200 -> 0.383314 -> 0.383578`

Example with CatBoost:

- `0.364996 -> 0.369634 -> 0.383434 -> 0.383453 -> 0.383885`

### 17.2 Interpretation
This is one of the strongest findings in the whole project:

- KMeans-derived features are the most important enhancement beyond the basic historical features
- Apriori is useful, but its marginal contribution is small
- ALS also helps, but only slightly

This means user segmentation is not just descriptive analysis. It materially improves the predictive model.

## 18. Segment-Level Evaluation
The supplementary notebook also evaluated model performance by user segment.

### 18.1 Which Segment Is Easiest to Predict
For every final model, Segment `1` is the easiest to predict by `F1@dynamic_k`.

Examples with ALS:

- CatBoost Segment 1: `F1@dynamic_k = 0.447281`
- LightGBM Segment 1: `0.445832`
- MLP Segment 1: `0.446647`
- Logistic Regression Segment 1: `0.432057`

### 18.2 Which Segment Is Hardest to Predict
For every final model, Segment `3` is the hardest to predict.

Examples with ALS:

- CatBoost Segment 3: `F1@dynamic_k = 0.330964`
- LightGBM Segment 3: `0.330427`
- MLP Segment 3: `0.330621`
- Logistic Regression Segment 3: `0.323228`

### 18.3 Interpretation
This tells us that different user types are not equally predictable. Some segments are naturally better aligned with the reorder framework, while others remain more difficult even for the best models.

## 19. Top-K Strategy Comparison
We compared three recommendation-length strategies:

- fixed `k = 11`
- rounded historical average basket size
- ceiling historical average basket size

### 19.1 Main Findings
Across all models, the same ranking appears:

- `rounded_avg_basket` gives the best F1
- `ceil_avg_basket` is second
- fixed `k = 11` is worst by F1

Examples:

- CatBoost: `0.352777 -> 0.382510 -> 0.383885`
- LightGBM: `0.352213 -> 0.381923 -> 0.383118`
- MLP: `0.352547 -> 0.382278 -> 0.383578`
- Logistic Regression: `0.342667 -> 0.369461 -> 0.370027`

### 19.2 Interpretation
Fixed `k = 11` produces higher recall and higher hit rate because it recommends more items, but it sacrifices precision. The personalized dynamic-k strategies achieve a better precision-recall balance, so they produce higher F1.

This is an important recommendation-level conclusion:

- the best ranking model is not enough by itself
- recommendation length policy matters
- grocery recommendation benefits from user-specific list sizes

## 20. What the Project Learned About the Domain
The strongest domain insights are:

1. Grocery reorders are heavily driven by habit and recency.
2. User heterogeneity matters enough that segmentation improves the model.
3. Basket context exists, but it is weaker than direct user-product history.
4. Collaborative signals help, but historical behavioral features explain most of the predictive power.
5. Personalized recommendation length is more effective than one fixed Top-K size for everyone.

## 21. Why the Final Results Make Sense
The final performance pattern is coherent with the feature design:

- Logistic Regression is too simple to fully exploit the nonlinear structure.
- LightGBM works very well because the task is tabular and rich in engineered numerical features.
- MLP performs especially well because the feature space is already informative and nonlinear.
- CatBoost remains strong, but because our final table is dominated by high-quality numerical interaction features rather than many raw categorical variables, it does not dramatically dominate the others.

## 22. Limitations
The project still has several limitations:

- candidate generation only includes previously purchased products, so full cold-start recommendation is not addressed
- the model is optimized mainly for reorder prediction, not for brand-new product discovery
- ALS gains are small, suggesting that richer collaborative or sequential methods may be needed for larger improvements
- no formal statistical significance testing was applied across folds
- recommendation calibration and threshold tuning could still be explored more deeply

## 23. Final Conclusions
The final project conclusions are:

1. A candidate-level supervised learning framework is effective for grocery reorder prediction.
2. The most important predictive signals are user-product interaction strength and recency.
3. KMeans-based segmentation adds meaningful downstream value and is the most important enhancement beyond the core behavioral features.
4. Apriori basket-context features and ALS similarity features both improve performance, but only marginally.
5. MLP with ALS is the best overall final model on recommendation-oriented metrics.
6. CatBoost with ALS has the highest AUC.
7. Dynamic-k based on historical basket size is a better recommendation strategy than a fixed Top-11 list.

## 24. Main Output Files
Key final files in the repository are:

- [build_model_dataset.py](/Users/bettinabopeng/Documents/GitHub/5971/build_model_dataset.py)
- [final_model_full_run.ipynb](/Users/bettinabopeng/Documents/GitHub/5971/final_model_full_run.ipynb)
- [supplementary_model_analysis.ipynb](/Users/bettinabopeng/Documents/GitHub/5971/supplementary_model_analysis.ipynb)
- [model_dataset_full.parquet](/Users/bettinabopeng/Documents/GitHub/5971/artifacts/model_dataset_full.parquet)
- [model_dataset_full.metadata.json](/Users/bettinabopeng/Documents/GitHub/5971/artifacts/model_dataset_full.metadata.json)
- [final_model_full_run_summary.csv](/Users/bettinabopeng/Documents/GitHub/5971/artifacts/final_model_full_run_summary.csv)
- [final_model_als_delta_summary.csv](/Users/bettinabopeng/Documents/GitHub/5971/artifacts/final_model_als_delta_summary.csv)
- [catboost_tuning_no_als.csv](/Users/bettinabopeng/Documents/GitHub/5971/artifacts/catboost_tuning_no_als.csv)
- [supplementary_ablation_summary.csv](/Users/bettinabopeng/Documents/GitHub/5971/artifacts/supplementary_ablation_summary.csv)
- [supplementary_segment_summary.csv](/Users/bettinabopeng/Documents/GitHub/5971/artifacts/supplementary_segment_summary.csv)
- [supplementary_segment_als_delta.csv](/Users/bettinabopeng/Documents/GitHub/5971/artifacts/supplementary_segment_als_delta.csv)
- [supplementary_topk_strategy_summary.csv](/Users/bettinabopeng/Documents/GitHub/5971/artifacts/supplementary_topk_strategy_summary.csv)
- [supplementary_segment_topk_summary.csv](/Users/bettinabopeng/Documents/GitHub/5971/artifacts/supplementary_segment_topk_summary.csv)
