# Instacart Grocery Reorder Prediction with User Segmentation, Apriori Rules, and ALS Features

## Team
Team 404 not Found

## 1. Executive Summary
This document is a **3/4 progress report** rather than the final report. Our project focuses on **next-order product prediction** on the Instacart Online Grocery Dataset: given what a user has purchased before, we predict which products they are most likely to buy in their next order.

Compared with our midterm presentation, this stage includes three major advances:

1. We provide stronger justification for **why K-Means was chosen** and **how the Apriori thresholds were selected**.
2. We complete a much richer feature engineering pipeline by adding **segmentation features, association-rule features, and ALS similarity features**.
3. We conduct a broader model comparison across **Logistic Regression, LightGBM, MLP, and CatBoost**, and explicitly compare **with ALS** versus **without ALS**.

The main takeaway at this stage is that:

- all nonlinear models outperform Logistic Regression
- MLP + ALS currently gives the strongest overall recommendation performance
- CatBoost + ALS gives the highest AUC
- ALS helps all models slightly, but its gains are modest
- the most important predictive signals still come from **user-product interaction and recency**

## 2. What Is New Since Midterm
Relative to the midterm report, this version adds the following:

### 2.1 More Complete Method Justification
At midterm, K-Means user segmentation and Apriori association analysis were presented as promising ideas. In this version, we explain in more detail:

- why K-Means is reasonable for this user behavior space
- why `K = 4` was chosen instead of larger or smaller values
- why aisle-level Apriori is preferred over product-level Apriori
- why the final Apriori thresholds were selected through sensitivity and tradeoff analysis

### 2.2 More Mature Feature Engineering
The current version goes beyond basic historical statistics. We now include:

- user-product interaction features
- user-level features
- product-level features
- K-Means segmentation features
- Apriori rule-based features
- ALS similarity features

### 2.3 Broader Empirical Comparison
The midterm version mainly motivated the pipeline. This version adds a full model comparison and a direct **ALS vs. no-ALS** experiment.

## 3. Business Motivation
The project addresses a recommendation problem with direct business relevance. Predicting the next order can support:

- product recommendation
- basket completion
- cross-sell and bundling
- user-specific recommendation strategies

The grocery setting is especially suitable for this because it contains both:

- **habitual repeat behavior**, where users reorder the same products frequently
- **contextual co-purchase behavior**, where products are bought together because of basket context

## 4. Dataset
We use the **Instacart Online Grocery Dataset**, which contains roughly:

- 3.4 million orders
- more than 33 million product-level purchase records
- about 206 thousand users

The raw tables include:

- `orders`
- `order_products__prior`
- `order_products__train`
- `products`
- `aisles`
- `departments`

## 5. Task Reformulation and Data Construction
One major difficulty in the raw Instacart dataset is that the original Kaggle train/test split is **not directly usable for standard supervised learning**. The reason is that the final order for test users does not include product-level labels.

To solve this, we redefine the task at the user level:

- for each user, the **last order with product-level information** is treated as the target order
- all earlier orders are treated as historical behavior

This converts the problem into a standard **next-order prediction** setting.

We then build candidate rows at the `(user, product)` level:

- if the product appears in the target order, `label = 1`
- otherwise, `label = 0`

This gives us a supervised modeling table for reorder prediction.

### 5.1 Cleaning
- We remove invalid `"missing"` categories from the product hierarchy.
- We fill `days_since_prior_order` missing values with `0`, because these correspond to first orders rather than random dirty data.

### 5.2 Why This Reformulation Matters
This redesign lets us:

- keep a large amount of user history
- avoid discarding users unnecessarily
- build a consistent supervised learning problem
- evaluate recommendation performance in a realistic setting

## 6. Key EDA Findings
Our exploratory analysis produced three central insights.

### 6.1 User Heterogeneity
Users differ substantially in:

- number of total orders
- number of unique products purchased
- shopping breadth
- reorder behavior

This motivates user segmentation.

### 6.2 Long-Tail Basket and Product Behavior
Both basket size and user-product purchase frequency show strong long-tail structure. A small number of user-product pairs are repeatedly purchased many times, while most are relatively sparse.

### 6.3 Strong Recency Effect
The most important EDA finding is the **recency effect**: the more recently a product was purchased, the more likely it is to appear again in the target order. This strongly supports including recency-based features in the final model.

### 6.4 Why the `orders_since_last` Bin Plot Matters
The `orders_since_last` binned line plot summarizes the relationship between recency and reorder probability. The x-axis measures how many orders have passed since the user last bought a product, and the y-axis measures the average probability that the product appears again in the target order.

The downward trend shows a clear monotonic pattern: recently purchased products are much more likely to be reordered, while products not purchased for many orders have much lower reorder probability. We use bins because the raw data are too large and noisy to visualize directly, so binned averages make the trend interpretable.

## 7. User Segmentation with K-Means
To capture shopping-style heterogeneity, we cluster users using **K-Means**.

### 7.1 Features Used for Clustering
We selected five user-level features that represent different behavioral dimensions:

- total historical orders
- average days between orders
- average basket size
- reorder rate
- number of unique products purchased

These features describe:

- activity level
- purchase rhythm
- order scale
- loyalty
- exploration breadth

### 7.2 Why K-Means
We chose K-Means for three main reasons:

- it is computationally efficient for a large dataset
- it produces clusters that are relatively easy to interpret
- it is suitable for our low-dimensional behavioral feature space

We also examined the structure of the user feature space. PCA shows that the first two principal components explain about **76.5%** of the variance, which suggests that user behavior has meaningful low-dimensional structure.

### 7.3 Why `K = 4`
We tested `K = 2` through `K = 9`. The inertia decreases continuously, but there is no strong elbow point. The silhouette score is highest at `K = 2` and then decreases, suggesting that the data are closer to a **continuous behavioral space** than to naturally separated clusters.

Therefore, we selected `K = 4` as a compromise between:

- statistical compactness
- business interpretability

This means we are not claiming that the users form four perfectly separated natural classes. Instead, we are using four clusters because they provide the most useful segmentation for downstream modeling and interpretation.

### 7.4 Why Not DBSCAN
We also considered DBSCAN, but it was not suitable in this context. In our experiments:

- one dominant cluster absorbed most users
- very small minor clusters appeared
- about 23% of users were classified as noise

So for this large-sample, low-dimensional, continuous feature space, K-Means was more stable and more practical.

### 7.5 Business Meaning of the Segments
The final four segments are broadly interpretable as:

- heavy users
- exploratory users
- light users
- routine replenishment users

These segments are not only descriptive; they are also integrated into the model through cluster dummy variables and cluster-based interaction features.

## 8. Basket-Level Association Analysis with Apriori
To study co-purchase behavior, we apply **Apriori** at the **aisle level** rather than the product level.

### 8.1 Why Aisle-Level Instead of Product-Level
We chose aisle-level association analysis because product-level combinations are too sparse and unstable. The aisle level provides:

- less sparsity
- more stable co-purchase patterns
- more interpretable cross-sell signals

### 8.2 Rule Metrics
We use three standard Apriori metrics:

- **support**: how often the combination appears overall
- **confidence**: how likely the consequent appears given the antecedent
- **lift**: how much stronger the co-purchase pattern is than random expectation

Their definitions are:

**Support**

`support(A, B) = #(A, B) / #(all baskets)`

Here, `#(A, B)` means the number of baskets containing both aisle `A` and aisle `B`. Support tells us how frequent a co-purchase pair is in the full dataset.

**Confidence**

`confidence(A -> B) = support(A, B) / support(A) = P(B | A)`

Confidence measures the conditional probability that aisle $B$ appears when aisle $A$ already appears.

**Lift**

`lift(A -> B) = P(A, B) / (P(A) * P(B))`

Lift compares the observed co-occurrence against random expectation. If lift is greater than 1, the two aisles co-occur more often than would be expected by chance.

### 8.3 Why These Thresholds
We selected the final thresholds by balancing **rule quality** and **rule coverage**. Very loose thresholds produced too many noisy rules, while very strict thresholds produced too few rules. Based on the sensitivity table and tradeoff plots, we selected:

- `support = 0.01`
- `confidence = 0.30`
- `lift = 1.30`

This combination gives a moderate number of aisle pairs while maintaining reasonably strong median lift.

### 8.4 How Apriori Enters the Model
We do not use association rules as a standalone recommender. Instead, we convert them into a feature:

- `apriori_rule_hits`

This feature indicates whether a candidate product’s aisle is supported by the user’s recent basket context. In this way, Apriori provides cross-sell information that complements the machine learning model.

## 9. Feature Engineering
Our current feature system includes six main groups.

### 9.1 User-Product Interaction Features
These are the most important features in the whole project. They include:

- purchase frequency
- reorder ratio
- recency in orders
- recency in days
- first and last order positions
- average add-to-cart position

These features capture habitual purchasing behavior directly.

### 9.2 Product-Level Features
These describe overall product behavior in the population:

- product reorder ratio
- total product purchases
- number of unique users buying the product
- average add-to-cart position

### 9.3 User-Level Features
These describe overall user behavior:

- total orders
- reorder rate
- total purchased items
- average basket size
- average days between orders
- number of unique products purchased

### 9.4 Segmentation Features
These include:

- user cluster dummy variables
- `cluster_product_target_rate`

They help the model balance a user’s own history against broader segment-level trends.

### 9.5 Apriori Feature
- `apriori_rule_hits`

This adds basket-context information.

### 9.6 ALS Features
We also compute collaborative-filtering-style similarity features:

- `als_dot`
- `als_cos`

These are used to test whether latent user-item similarity improves prediction beyond the engineered features.

The ALS part is based on **Alternating Least Squares**, a matrix factorization method for recommender systems. The main idea is to factorize the user-item interaction matrix into:

- a latent vector `u_i` for each user
- a latent vector `v_j` for each product

so that the observed interaction strength is approximated by:

`R_ij ~= u_i^T v_j`

where `R_ij` is the historical interaction between user `i` and product `j`.

ALS alternates between two steps:

- fixing item vectors and solving for user vectors
- fixing user vectors and solving for item vectors

until the reconstruction objective is optimized.

From these latent vectors, we construct two additional features:

**ALS dot product**

`als_dot_ij = u_i^T v_j`

This measures raw latent affinity between a user and a product.

**ALS cosine similarity**

`als_cos_ij = (u_i^T v_j) / (||u_i|| * ||v_j||)`

This normalizes the affinity and measures directional similarity between the user and product embeddings.

In our framework, ALS is not used as the final recommender by itself. Instead, it provides supplementary similarity signals that are fed into the supervised models.

### 9.7 Perfect-Correlation Cleanup
During feature analysis, we found that:

- `up_buy_cnt`
- `up_reorder_cnt`

were perfectly correlated in the final modeling table. Therefore, we:

- kept `up_buy_cnt`
- removed `up_reorder_cnt`

This ensures that the final feature set avoids redundant perfectly correlated information.

## 10. Models and How They Work
We compare four models.

### 10.1 Logistic Regression
This is our linear baseline. It predicts reorder probability as a weighted linear combination of input features. It is simple and interpretable, but it cannot naturally capture complex nonlinear relationships or rich feature interactions.

More formally, Logistic Regression models the reorder probability as:

`P(y = 1 | x) = sigmoid(w^T x + b)`

where:

- `x` is the feature vector
- `w` is the model coefficient vector
- `b` is the bias term
- `sigmoid(.)` is the sigmoid function

The sigmoid function is:

`sigmoid(z) = 1 / (1 + exp(-z))`

This converts the linear score into a probability between 0 and 1.

### 10.2 LightGBM
LightGBM is a gradient boosting decision tree model. It builds many trees sequentially, where each new tree corrects the mistakes of the previous ones. It is very strong on structured tabular data and is especially good at capturing nonlinearities and feature interactions.

The general boosting form can be written as:

`F_M(x) = sum_{m=1 to M} gamma_m * h_m(x)`

where:

- `h_m(x)` is the `m`-th decision tree
- `gamma_m` is its contribution weight
- `F_M(x)` is the final prediction function

Each new tree is trained to reduce the residual errors or gradient direction left by the current ensemble. LightGBM is computationally efficient because it uses histogram-based splitting and leaf-wise tree growth, which is well suited for large tabular data.

### 10.3 MLP
MLP, or multilayer perceptron, is a feed-forward neural network with hidden layers. It learns nonlinear mappings between features and the target, making it more flexible than Logistic Regression. In our setting, it is able to leverage the engineered features very effectively.

For one hidden layer, the computation can be expressed as:

`h = phi(W1 x + b1)`

`y_hat = sigmoid(W2 h + b2)`

where:

- `x` is the input feature vector
- `W1, W2` are weight matrices
- `b1, b2` are bias terms
- `phi(.)` is a hidden-layer activation function
- `sigmoid(.)` is the output sigmoid

Because of these nonlinear hidden transformations, MLP can learn more complex decision boundaries than linear models.

### 10.4 CatBoost
CatBoost is another boosting-based model. It is particularly strong when categorical structure matters, and it can handle mixed tabular data very effectively. In our project, it remains very competitive, although it does not dramatically outperform MLP or LightGBM.

Like LightGBM, CatBoost is also an ensemble of boosted decision trees:

`F_M(x) = sum_{m=1 to M} gamma_m * h_m(x)`

Its main strength lies in robust handling of categorical information and in reducing prediction bias during training through ordered boosting. In our project, CatBoost works on a feature table already dominated by strong numerical interaction features, so its advantage over LightGBM and MLP is present but not dramatic.

## 11. Experimental Design
We evaluated the models under two feature settings:

- **no_als**: base features only
- **with_als**: base features plus ALS similarity features

We report both row-level and recommendation-level metrics.

### 11.1 Why These Metrics
We use:

- `AUC`
- `PR-AUC`
- `LogLoss`
- `F1@11`
- `F1@dynamic_k`

The reorder problem is imbalanced, so `PR-AUC` is especially important. For recommendation quality, `F1@dynamic_k` is one of the most relevant metrics because the list length is adapted to the user’s historical average basket size.

The calculation principles are as follows.

#### AUC
AUC stands for the **Area Under the ROC Curve**. The ROC curve plots:

- true positive rate
- false positive rate

across different thresholds.

The true positive rate is:

`TPR = TP / (TP + FN)`

The false positive rate is:

`FPR = FP / (FP + TN)`

AUC measures how well the model ranks positive samples above negative samples independent of a single threshold.

#### PR-AUC
PR-AUC is the **Area Under the Precision-Recall Curve**.

Precision is:

`Precision = TP / (TP + FP)`

Recall is:

`Recall = TP / (TP + FN)`

Since reorder prediction is imbalanced, PR-AUC is especially important because it evaluates how well the model captures positives while controlling false positives.

#### LogLoss
LogLoss evaluates the quality of predicted probabilities:

`LogLoss = -(1 / N) * sum_i [ y_i * log(p_hat_i) + (1 - y_i) * log(1 - p_hat_i) ]`

where:

- `y_i` is the true label
- `p_hat_i` is the predicted probability

Lower LogLoss means the predicted probabilities are closer to the true outcomes.

#### F1@11
`F1@11` is the F1 score computed after recommending the **top 11 products** for each user.

F1 is:

`F1 = 2 * Precision * Recall / (Precision + Recall)`

This metric evaluates recommendation quality under a fixed-length recommendation list.

#### F1@dynamic_k
`F1@dynamic_k` uses the same F1 definition, but the list length `k` is dynamic and determined by the user’s historical average basket size. This is more realistic in grocery shopping because users do not all buy the same number of products.

#### Hit@k
Although not always emphasized as the main ranking metric, we also compute Hit@k. It checks whether at least one of the recommended products is correct:
`Hit@k = 1 if at least one true item appears in the top-k list, otherwise 0`

The final value is averaged across users.

### 11.2 Why Compare ALS and No ALS
The point of the ALS comparison is not to replace our main feature engineering pipeline, but to test whether latent user-item similarity provides complementary information. This helps us answer whether collaborative filtering signals add value on top of the historical interaction features.

## 12. Current Results
These results should be interpreted as **current 3/4-stage results**, not final locked results. Their purpose is to show that the end-to-end pipeline is working and already producing stable, comparable outcomes.

### 12.1 Mean Cross-Validated Performance

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

### 12.2 How to Read These Results
There are three clear patterns.

First, all nonlinear models outperform Logistic Regression, which indicates that reorder prediction contains important nonlinear relationships.

Second, LightGBM, MLP, and CatBoost are all very strong and quite close to one another. If we focus on recommendation-oriented metrics, **MLP + ALS** is currently the strongest model overall. If we focus only on AUC, **CatBoost + ALS** is slightly higher.

Third, ALS helps, but only a little. The gains are positive and consistent, but they are not dramatic.

### 12.3 ALS Improvement

| Model | Δ AUC | Δ PR-AUC | Δ F1@11 | Δ F1@dynamic_k | Δ LogLoss |
|---|---:|---:|---:|---:|---:|
| CatBoost | +0.000710 | +0.000293 | +0.000406 | +0.000155 | -0.000693 |
| LightGBM | +0.000730 | +0.000472 | +0.000489 | +0.000513 | -0.000311 |
| LogisticRegression_SAGA | +0.000534 | +0.000136 | +0.000202 | +0.000194 | -0.000473 |
| MLP | +0.000702 | +0.000432 | +0.000237 | +0.000213 | -0.000234 |

Interpretation:

- ALS improves all models on average.
- the gain is real, but modest
- the largest `F1@dynamic_k` improvement appears in LightGBM
- the strongest overall current model remains MLP + ALS

## 13. Discussion

### 13.1 Why Logistic Regression Underperforms
Logistic Regression is useful as a baseline, but this task contains nonlinear interactions among recency, interaction strength, product repeatability, and user heterogeneity. A linear model is not expressive enough to fully capture these relationships.

### 13.2 Why CatBoost Does Not Dramatically Dominate
The literature often reports strong CatBoost results on tabular problems, but our task is heavily driven by **high-quality numerical interaction features**, not only by raw categorical variables. In this setting:

- LightGBM already handles the tabular structure very well
- MLP can exploit the engineered nonlinear feature space effectively
- CatBoost remains competitive, but does not dominate by a wide margin

### 13.3 Why ALS Helps Only a Little
ALS contributes a collaborative similarity signal, but the problem is already strongly explained by:

- user-product interaction history
- recency signals
- product reorder statistics
- segmentation features
- Apriori context

So ALS acts as a complement rather than the main driver of performance.

## 14. Main Conclusions At The 3/4 Stage
At the current stage, we can conclude that:

1. The Instacart next-order prediction problem can be effectively modeled as candidate-level supervised learning.
2. The strongest predictive signals come from user-product interaction and recency.
3. K-Means segmentation and Apriori aisle-level rules provide useful context beyond basic historical statistics.
4. ALS features improve performance consistently, but only by a small margin.
5. At the current 3/4 stage, **MLP + ALS** is the strongest overall model by the most relevant recommendation metrics, while CatBoost and LightGBM remain very close alternatives.

## 15. Limitations
- Candidate generation only uses historically purchased products, so full cold-start recommendation is not addressed.
- ALS gains are small, suggesting that richer collaborative signals may be needed for larger improvements.
- We did not perform a formal statistical significance test across folds.
- CatBoost tuning was intentionally constrained to stay aligned with the earlier no-ALS tuning logic.
- This is still a progress report rather than the final polished deliverable.

## 16. Future Work Before The Final Report
- improve visualization and presentation quality
- prepare the poster and demo
- perform feature ablation more systematically
- analyze model performance by user segment
- explore richer candidate expansion and ranking-oriented objectives

## 17. Output Files
The main experiment outputs are stored in:

- [final_model_full_run_summary.csv](/Users/bettinabopeng/Documents/GitHub/5971/artifacts/final_model_full_run_summary.csv)
- [final_model_als_delta_summary.csv](/Users/bettinabopeng/Documents/GitHub/5971/artifacts/final_model_als_delta_summary.csv)
- [catboost_tuning_no_als.csv](/Users/bettinabopeng/Documents/GitHub/5971/artifacts/catboost_tuning_no_als.csv)
- [model_dataset_full.parquet](/Users/bettinabopeng/Documents/GitHub/5971/artifacts/model_dataset_full.parquet)

## Appendix: Q&A

### 1. How distinct are these four segments in practice? Are the clusters well separated, or do some of them overlap substantially in behavior?
The four segments are not completely isolated groups with sharp boundaries. Instead, they represent relatively stable behavioral patterns identified from user-level features such as activity, purchase rhythm, basket size, reorder tendency, and exploration breadth. In practice, the clusters are interpretable and distinguishable, but they do overlap to some extent. Our analysis suggests that user behavior is more continuous than perfectly partitioned, so the choice of four clusters is mainly a balance between statistical structure and business interpretability.

### 2. How do you deal with class imbalance in reorder prediction?
Class imbalance is an important issue in this task, since the positive rate is only around 10 percent. We address it in several ways. First, we restrict the candidate set to products the user has already purchased historically, which reduces extreme imbalance compared with using the full product catalog. Second, for Logistic Regression we use `class_weight="balanced"`. Third, for tree-based models such as LightGBM and CatBoost, we use class weighting schemes like `scale_pos_weight`. Finally, instead of relying on accuracy, we focus on metrics that are more suitable for imbalanced recommendation problems, such as `PR-AUC`, `F1@11`, and `F1@dynamic_k`.

### 3. Why was MLM necessary in your framework instead of using only traditional prediction models?
In our framework, MLM refers to the full machine learning pipeline rather than just a single prediction model. A traditional classifier only gives us purchase probabilities, but a recommendation system requires more than that. We need a structured process for candidate generation, feature construction, probability estimation, ranking, and Top-K recommendation output. MLM was necessary because our final objective is not simply classification, but producing a practical recommendation list for the next basket.

### 4. Apart from logistic regression, what were the other two models you plan to use and how do they work?
The two main additional models were **LightGBM** and **MLP**, and we also extended the comparison to **CatBoost**. LightGBM is a gradient boosting decision tree model that works very well on structured tabular data and can capture nonlinear effects and feature interactions. MLP, or multilayer perceptron, is a feed-forward neural network that learns nonlinear mappings through hidden layers. CatBoost is another boosting-based model that is particularly strong when categorical information is important. In our task, all three nonlinear models outperformed Logistic Regression.

### 5. How do you distinguish between true user preference and habitual purchasing behavior in your reorder predictions?
We do not separate those two concepts with a strict hard boundary, but instead approximate them through different groups of features. Habitual purchasing behavior is mainly captured by user-product interaction features such as `up_buy_cnt`, `up_freq`, `up_reorder_ratio`, `up_orders_since_last`, and `up_days_since_last`. More stable preference patterns are reflected by user-level and segment-level features, such as `u_unique_products`, `u_reorder_ratio`, user cluster dummies, and `cluster_product_target_rate`. So the model sees both habit-driven signals and broader preference structure, then learns how to weigh them jointly.

### 6. How do you integrate association rules with your machine learning model?
We do not use association rules as a standalone recommendation engine. Instead, we transform them into an additional feature. We first run Apriori at the aisle level to identify strong co-purchase patterns. Then, for each user, we examine the aisles in the most recent historical basket. If a candidate product’s aisle is supported by one of those rules, we assign a positive value to the feature `apriori_rule_hits`. In this way, association rules provide cross-sell and bundling signals that complement the machine learning model.

### 7. What features were most important for predicting reorders?
The most important features were the user-product interaction features, especially `up_freq`, `up_buy_cnt`, `up_reorder_ratio`, `up_orders_since_last`, and `up_days_since_last`. These variables directly capture how often the user bought the product and how recently they bought it, which are central signals in reorder prediction. Product-level repeatability, user-level activity measures, and cluster-level features were also useful, while ALS and Apriori features provided smaller but still positive contributions.

### 8. How do you handle users with very sparse interaction history in the segmentation approach?
Sparse-history users are naturally more difficult to characterize, but our segmentation design is based on aggregated behavioral statistics rather than long sequential histories. This makes the approach more robust. We use features such as total orders, average basket size, average days between orders, reorder rate, and number of unique products purchased. Missing or weak history is handled through computable default statistics and stable aggregation. In practice, sparse users tend to be assigned to lighter, lower-activity segments rather than to highly specialized groups.

### 9. How does your model balance a user's own past habits with the general trends of their segment?
That balance is achieved at the feature level. The user’s own purchase history is primarily captured by the `up_*` interaction features, which are the strongest predictors in the model. Segment-level trends are represented by features such as `user_cluster_*` and `cluster_product_target_rate`. In practice, the model relies more heavily on the user’s own history, because reorder behavior is highly personalized. Segment-level information acts as a secondary adjustment, especially when individual history is weaker or when broader behavioral patterns are informative.

### 10. Why do you choose KMeans while doing user clustering, and what are the limitations in this context? Do you consider using other k values?
We chose KMeans mainly because it is computationally efficient, scalable to a large dataset, and produces clusters that are relatively easy to interpret. This matches our goal of building usable business-oriented user segments. We did evaluate multiple values of `K`, from 2 to 9. Although inertia decreased steadily and there was no perfectly clear elbow, `K = 4` gave the best balance between statistical separation and interpretability. The limitations are that KMeans assumes relatively compact cluster shapes, depends on scaling and the choice of `K`, and is less natural when the data distribution is continuous rather than strongly clustered. We also considered DBSCAN, but in our data it produced highly imbalanced clusters and labeled too many points as noise.

### 11. How do you explain the `orders_since_last` bin lines?
The `orders_since_last` binned line plot shows the relationship between recency and reorder probability. The x-axis represents how many orders have passed since the user last purchased a product, and the y-axis represents the average probability that the product appears again in the target order. The overall downward pattern shows a clear recency effect: products bought more recently are much more likely to be reordered, while products not purchased for many orders have a much lower chance of reappearing. We use bins because the raw data are large and noisy, so binned averages make the trend much easier to interpret. This plot confirms that recency is one of the strongest predictors in our framework and supports the inclusion of features such as `up_orders_since_last` and `up_days_since_last`.
