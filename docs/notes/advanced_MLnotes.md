## Advanced Machine Learning Interview Questions

1. **What are the difficulties when dealing with right skewed data in modeling and statistical testing?**

   Dealing with **right-skewed data** (where the tail is longer on the right) presents several challenges in **modeling and statistical testing** due to its asymmetric distribution. Here are the key difficulties:

   **1. Violation of Normality Assumptions**

   - Many statistical tests (e.g., t-tests, ANOVA, linear regression) assume normally distributed residuals. Right-skewed data violates this assumption, leading to:
     - Biased estimates
     - Incorrect p-values and confidence intervals
     - Reduced power of hypothesis tests

   **2. Impact on Mean and Variance**

   - The **mean > median** in right-skewed data, making the mean sensitive to extreme values.
   - High skewness inflates variance, affecting models that assume homoscedasticity (constant variance).

   **3. Poor Performance of Parametric Models**

   - **Linear regression** and other parametric models perform poorly when errors are non-normal.
   - Machine learning models (e.g., linear regression, SVM) may underperform if not adjusted for skewness.

   **4. Misleading Summary Statistics**

   - The **mean** is pulled toward the tail, often misrepresenting central tendency.
   - **Standard deviation** becomes less meaningful due to outliers.

   **5. Challenges in Statistical Testing**

   - Non-parametric tests (e.g., Mann-Whitney U, Kruskal-Wallis) may be needed instead of t-tests/ANOVA.
   - Log transformation (or Box-Cox) is often required to normalize data before analysis.

   **6. Machine Learning Model Biases**

   - Tree-based models (e.g., Random Forest, XGBoost) handle skewness better, but linear models (e.g., OLS) suffer.
   - Models optimizing **MSE** (Mean Squared Error) are disproportionately affected by large right-tail values.

   **7. Outlier Sensitivity**

   - Right-skewed data often contains extreme values, making models prone to overfitting.
   - Robust scaling (e.g., median/IQR) may be needed instead of standardization (mean/SD).

   **Solutions to Handle Right-Skewed Data**

   1. **Transformations**:
      - **Log transformation** (if data is positive and non-zero).
      - **Square root** or **Box-Cox transformation**.
   2. **Non-parametric tests** (if normality cannot be achieved).
   3. **Robust regression methods** (e.g., quantile regression, Huber regression).
   4. **Binning or Categorization** (if appropriate for the problem).
   5. **Using Median instead of Mean** for central tendency.

2. **Consequences of right-skewness in independent variables and dependent variable respectively**

   The challenges differ depending on whether the **independent variables (features/predictors)** or the **dependent variable (target/response)** is right-skewed. Each scenario introduces distinct problems in **modeling and exploratory analysis**.

   **1. Right-Skewed Independent Variables (Predictors)**

   **Problems in Modeling:**

   - **Biased Coefficient Estimates:** Skewed predictors can distort regression coefficients, especially in **linear models (OLS, GLM)**, leading to unreliable interpretations.
   - **Heteroscedasticity (Non-Constant Variance):**  
     - Extreme values can cause unequal error variances, violating OLS assumptions.
   - **Poor Model Performance:** Algorithms like **linear regression, logistic regression, and SVM** (with linear kernels) may perform poorly if predictors are highly skewed.
   - **Overfitting to Outliers:** Models may assign excessive weight to extreme values, reducing generalization.

   **Problems in Exploratory Analysis (EDA):**

   - **Misleading Correlations:** Pearson correlation assumes linearity and can be inflated/deflated by skewness.
   - **Biased Feature Importance:** Tree-based models (e.g., Random Forest) may overemphasize skewed features due to splits on extreme values.
   - **Visualization Challenges:** Histograms and boxplots may be dominated by outliers, obscuring true patterns.

   **Solutions:**  

   - **Log/Box-Cox transformation** (if feature is positive).  
   - **Robust scaling** (e.g., using median/IQR instead of mean/SD).  
   - **Binning or quantile-based discretization**.  
   - **Non-linear models (e.g., decision trees, neural networks)** that handle skewness better.  

   **2. Right-Skewed Dependent Variable (Target)**

   **Problems in Modeling:**

   - **Non-Normal Residuals:** Linear regression assumes normally distributed residuals; skewness violates this, leading to biased **p-values and confidence intervals**.  
   - **Poor Prediction Accuracy:** Models optimizing **MSE (Mean Squared Error)** are heavily influenced by extreme values, worsening predictions for typical cases.  
   - **Incorrect Loss Function Optimization:** If the target is highly skewed (e.g., income, insurance claims), a model minimizing MSE will be pulled toward overpredicting high values.  

   **Problems in Exploratory Analysis (EDA):**

   - **Mean Misrepresents Central Tendency:** The mean is inflated by the right tail, making the **median** a better measure.  
   - **Difficulty in Detecting Patterns:** Scatterplots may show a "fan shape" due to increasing variance with larger values.  
   - **Class Imbalance Analog (for Regression):** Similar to classification, rare high-value cases dominate loss functions, leading to poor performance on typical values.  

   **Solutions:**  

   - **Log-transform the target** (if strictly positive) to make distribution more symmetric.  
   - **Use alternative loss functions** (e.g., **MAE, Huber loss, or quantile regression**).  
   - **Tree-based models (e.g., XGBoost, Random Forest)** that are less sensitive to skewness.  
   - **Gamma or Tweedie regression** (for strictly positive, right-skewed targets like insurance claims).  

   **Key Differences Summary**

   | **Aspect**                | **Right-Skewed Independent Variables**      | **Right-Skewed Dependent Variable**             |
   | ------------------------- | ------------------------------------------- | ----------------------------------------------- |
   | **Main Modeling Issue**   | Biased coefficients, heteroscedasticity     | Non-normal residuals, poor MSE optimization     |
   | **Exploratory Challenge** | Misleading correlations, outlier distortion | Mean misrepresentation, fan-shaped residuals    |
   | **Best Fixes**            | Log transform, robust scaling, tree models  | Log target, MAE/Huber loss, quantile regression |

3. **How does tree-based model handle right-skewness in independent variables and dependent variable respectively?**

   **1. Tree-Based Models and Skewed *Independent* Variables (Features)**

   **Why They Can Overemphasize Skewed Features:**

   - **Split Behavior:** Decision trees make splits by finding thresholds that best separate data. If a feature is highly right-skewed (e.g., income, where most values are low but a few are extremely high), the tree might:
     - Create splits at extreme values (e.g., `income > $1M`), leading to **sparse or unbalanced branches**.
     - Over-prioritize the skewed feature because it appears to offer high information gain (due to extreme values), even if other features are more meaningful.
   - **Result:** The model may **overfit to rare extreme values**, reducing generalization.

   **Why They’re Still Suggested as a Solution?**

   - Unlike linear models, trees **do not assume normality or linearity**, so they can *handle* skewness better in the sense that they won’t fail outright (e.g., no assumptions violated).
   - However, they may still **perform suboptimally if skewness introduces noise or outliers**.  
   - **Solution:** Even when using trees, you might still want to:
     - Transform skewed features (e.g., log) to make splits more balanced.
     - Use regularization (e.g., limiting `max_depth` in Random Forest) to prevent overfitting to extremes.

   **2. Tree-Based Models and Skewed *Dependent* Variables (Target)**

   **Why They Handle Skewed Targets Well:**

   - **Loss Function Flexibility:**  
     - Trees minimize impurity (e.g., Gini, MSE for regression) and can adapt to skewed targets without assuming normality.  
     - For regression, using **MSE** on a right-skewed target still works (though it may be pulled by outliers), but alternatives like **MAE** or **quantile-based splits** can help.  
   - **No Distributional Assumptions:**  
     - Unlike linear regression (which assumes normal residuals), trees make no such assumptions, so skewness in the target is less problematic.  

   **When They Struggle:**  

   - If the target has **extreme outliers**, trees may still create splits that overfit to them (similar to skewed features).  
   - **Fix:** Use **log-transform on the target** or **quantile regression forests** to focus on median/robust predictions.  

   **Key Clarification**

   - **For skewed *features*:**  
     - Trees *can* handle skewness better than linear models (no normality needed), but extreme skew may still hurt performance.  
     - Preprocessing (e.g., log transform) is often **recommended even for trees** to avoid overemphasizing outliers.  

   - **For skewed *targets*:**  
     - Trees are **generally more robust** than linear models because they don’t assume normal residuals.  
     - However, extreme skew can still distort splits—transforming the target (e.g., log) or using robust loss functions helps. 

4. **What would you do when you want to perform hypothesis testing (e.g. student t-test) while the data has high kurtosis?**

   When your data has **high kurtosis** (heavy tails or outliers), the assumptions of standard parametric tests like the **Student’s t-test** (which assumes normality) may be violated. High kurtosis can lead to:

   1. **Inflated Type I error rates** (false positives) due to heavier tails than a normal distribution.
   2. **Reduced power** if the test is not robust to deviations from normality.

   **Solutions for Hypothesis Testing with High Kurtosis**

   **1. Check Normality More Rigorously**

      - Use **Shapiro-Wilk** or **Anderson-Darling** tests (though they are sensitive to sample size).
      - Visual checks: **Q-Q plots**, histograms, or kernel density plots.

   **2. Use a More Robust Test**

      - **Welch’s t-test**: Does not assume equal variances and is more robust to slight non-normality.
      - **Mann-Whitney U test (Wilcoxon rank-sum test)**: Non-parametric alternative to the t-test (does not assume normality but assumes similar distributions under \(H_0\)).
      - **Yuen’s t-test**: A robust version of the t-test that trims outliers (useful for heavy-tailed data).
      - **Permutation test**: Does not rely on distributional assumptions (resamples data to compute p-values).

   **3. Transform the Data**

      - Apply a **log, square root, or Box-Cox transformation** to reduce kurtosis (if data is positive and skewed).
      - Winsorization (capping extreme values) can help, but be cautious about modifying data.

   **4. Use Bootstrapping**

      - **Bootstrap the t-test**: Resample with replacement to estimate the sampling distribution and compute p-values empirically.
      - Example in R:
        ```R
        boot_t_test <- function(x, y, n_boot = 10000) {
          obs_diff <- mean(x) - mean(y)
          pooled <- c(x, y)
          boot_diffs <- replicate(n_boot, {
            new_x <- sample(pooled, length(x), replace = TRUE)
            new_y <- sample(pooled, length(y), replace = TRUE)
            mean(new_x) - mean(new_y)
          })
          p_value <- mean(abs(boot_diffs) >= abs(obs_diff))
          return(p_value)
        }
        ```

   **5. Bayesian Alternatives**

      - Use a **Bayesian t-test** with a robust likelihood (e.g., Student’s t-distribution instead of normal).

   **Recommendation**

   - **If sample size is large (n > 30)**: The t-test may still work reasonably well (CLT helps).
   - **If sample size is small and kurtosis is high**: Use **Yuen’s t-test, Mann-Whitney U, or bootstrapping**.
   - **Always report** that you checked kurtosis and why you chose an alternative method.

5. **What types of models require normalizing or standardizing data and why?**

   1. Distance-Based Algorithms: These algorithms rely on calculating distances between data points, making them sensitive to the scale of features.

      - **k-Nearest Neighbors (kNN)**: Uses Euclidean or Manhattan distance. Unscaled features can dominate the distance calculation.

      - **k-Means Clustering**: Relies on Euclidean distance to assign clusters. Uneven scales distort cluster shapes.

      - **Support Vector Machines (SVM)**: For linear or RBF kernels, unscaled features may bias the hyperplane or distance computations.

      **Why?** Features on larger scales dominate the distance metric, leading to biased results.

   2. Algorithms with Variance Assumptions: These assume that features contribute equally to the variance or are centered around zero.

      - **Principal Component Analysis (PCA)**: Maximizes variance along orthogonal axes. Unscaled features with high variance may artificially dominate PCA directions.

      - **Linear Discriminant Analysis (LDA)**: Assumes features are normally distributed and equally scaled.

      **Why?** PCA/LDA are sensitive to variance; unscaled features can skew the principal components or class separation.

   3. Models with Coefficient Penalty Terms: Regularization penalizes coefficients, and unscaled features lead to uneven penalties.

      - **Ridge/Lasso Regression (L1/L2 regularization)**: Penalizes large coefficients. Features on smaller scales may get unfairly suppressed.

      - **Logistic Regression (with regularization)**: Similar to linear models, coefficients are penalized based on scale.

      **Why?** Regularization treats all coefficients equally; unscaled features cause some coefficients to be penalized more than others.

   4. Gradient-Based Optimization: Algorithms using gradient descent converge faster with scaled data.

      - **Neural Networks**: Unscaled inputs cause unstable gradients and slow convergence.

      - **Linear/Logistic Regression (without closed-form solutions)**: Gradient descent suffers with uneven feature scales.

      **Why?** Different feature scales lead to ill-conditioned optimization landscapes, slowing convergence.

   5. Models That Do NOT Require Normalization/Standardization: These models are invariant to feature scales.

      - **Tree-Based Models (Decision Trees, Random Forest, Gradient Boosting)**: Splits are based on feature thresholds, not distances or coefficients.

      - **Naive Bayes**: Uses probability distributions (e.g., Gaussian NB handles scales internally).

      - **Rule-Based Models (e.g., RuleFit)**: Scale-independent logic.

      **Why?** Splitting rules in trees compare relative values, not absolute magnitudes. Scaling has no effect.
      
      **Summary Table**
      
      | **Model Type**               | **Needs Scaling?** | **Reason**                                                   |
      | ---------------------------- | ------------------ | ------------------------------------------------------------ |
      | kNN, k-Means, SVM (distance) | Yes                | Distance metrics are scale-sensitive.                        |
      | PCA, LDA                     | Yes                | Assumes equal variance contribution.                         |
      | Ridge/Lasso/Logistic (L1/L2) | Yes                | Regularization penalizes coefficients unfairly if unscaled.  |
      | Neural Networks              | Yes                | Gradient descent converges faster with scaled inputs.        |
      | Decision Trees/Random Forest | No                 | Splits are scale-invariant.                                  |
      | Gradient Boosting (XGBoost)  | No (but can help)  | Generally robust, but scaling may help in some cases (e.g., linear weak learners). |

6. **Write the pseudocode for hyperparameter tuning and k-fold cross validation from scratch.**

   ```
   INPUT:
       - X: Features
       - y: Labels/targets
       - model_class: A model class that can be instantiated with hyperparameters
       - param_grid: A dictionary of hyperparameters and their candidate values
       - k: Number of folds for cross-validation
       - scoring_function: A function to evaluate performance (e.g., accuracy, RMSE)
   
   OUTPUT:
       - best_params: Hyperparameter combination with best average score
       - best_score: The score corresponding to best_params
   
   FUNCTION k_fold_cross_validation(X, y, model_class, params, k, scoring_function):
       Split data X, y into k equal folds (shuffle before splitting)
       scores = []
   
       FOR i in 1 to k:
           validation_indices = fold i
           training_indices = all other folds
   
           X_train, y_train = data[training_indices]
           X_val, y_val = data[validation_indices]
   
           model = model_class(**params)
           model.fit(X_train, y_train)
   
           y_pred = model.predict(X_val)
           score = scoring_function(y_val, y_pred)
           scores.append(score)
   
       RETURN mean(scores)
   
   FUNCTION grid_search_cv(X, y, model_class, param_grid, k, scoring_function):
       best_score = -infinity
       best_params = None
   
       FOR each combination in param_grid:
           params = current combination
           avg_score = k_fold_cross_validation(X, y, model_class, params, k, scoring_function)
   
           IF avg_score > best_score:
               best_score = avg_score
               best_params = params
   
       RETURN best_params, best_score
   
   # Example usage (outside function scope)
   best_params, best_score = grid_search_cv(X, y, MyModel, param_grid, k=5, scoring_function=accuracy)
   ```

7. **What are the four assumptions of linear regression and what are the consequences if any of them is not met?**

   Linear regression has four key assumptions, and violating any of them can lead to problems like biased estimates, inefficient predictions, or invalid statistical inferences.

   To check the assumptions of linear regression, you need to fit the model first because most of the assumptions are about the residuals, which are only available *after* making predictions.

   | Assumption                    | Description                                                  | Consequences of Violation                                    |
   | ----------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
   | **1. Linearity**              | The relationship between the independent variables and the dependent variable is linear. | - Model underfits the data.- Predictions are systematically off.- Coefficients don't reflect actual relationships. |
   | **2. Independence**           | Residuals (errors) are independent — no autocorrelation.     | - Common in time series data.- Standard errors are underestimated → overconfident p-values and confidence intervals.- Inflated Type I errors. |
   | **3. Homoscedasticity**       | Constant variance of residuals across all levels of predictors. | - If violated (i.e., heteroscedasticity), the model gives inefficient estimates.- Standard errors become unreliable → misleading hypothesis tests. |
   | **4. Normality of Residuals** | Residuals should be normally distributed (especially for inference). | - Confidence intervals and hypothesis tests become invalid, especially in small samples.- Less impact on prediction, more on inference. |

   we can use residual diagnostics to check assumptions:

   | Assumption           | What to check/plot                                           |
   | -------------------- | ------------------------------------------------------------ |
   | **Linearity**        | Plot residuals vs. predicted values — should not show patterns. |
   | **Independence**     | Plot residuals over time/order — should not show trends. Use Durbin-Watson test for autocorrelation. |
   | **Homoscedasticity** | Same residuals vs. predicted plot — should show constant spread (no "fan" or "cone" shape). |
   | **Normality**        | Histogram or Q-Q plot of residuals — should look roughly normal. |

8. **How to detect multi-colinearity when performing linear regression?**

   Multicollinearity happens when two or more independent variables are highly correlated with each other. This makes it hard for the model to isolate the effect of each variable, leading to:

   - Unstable coefficients (they change a lot with small changes in data)
   - Inflated standard errors
   - Misleading p-values

   Main ways of detecting multicollinearity:

   1. correlation matrix: Check pairwise correlations between independent variables. If any correlation coefficient is close to +1 or -1 → potential multicollinearity.

   2. variance inflation factor (VIF): VIF quantifies how much the variance of a coefficient is inflated due to multicollinearity. 

      Rule of thumb: VIF > 5 → moderate multicollinearity; VIF > 10 → high multicollinearity (serious problem).

   How to fix multicollinearity:

   - Drop one of the correlated variables.
   - Combine them (e.g., PCA, feature engineering).
   - Use models that are less sensitive to multicollinearity (like tree-based models).

9. **Techniques to make your model robust to outliers** vs **Techniques to make your model robust to skewness?**

   Outliers and skewness are related but distinct challenges in statistical modeling and machine learning. 

   **Techniques for Outlier Robustness**

   1. Robust estimation methods:
      - Median instead of mean
      - Huber loss or quantile regression
      - RANSAC (Random Sample Consensus)
      - M-estimators that downweight extreme observations
   2. Data transformation/preprocessing:
      - Winsorization (capping extreme values)
      - Removing statistical outliers (e.g., beyond 3 standard deviations)
      - Using robust scaling (based on median and IQR instead of mean and standard deviation)
   3. Model choices:
      - Decision trees and tree-based models (inherently resistant to outliers)
      - Support Vector Machines with appropriate kernels
      - Using ensemble methods like Random Forest or Gradient Boosting
   4. Regularization to prevent the model from fitting to outliers

   **Techniques for Skewness Robustness**

   Skewness refers to asymmetry in the distribution of data. To handle skewness:

   1. Data transformations:
      - Log transformation for right-skewed data
      - Square root transformation for moderate right skewness
      - Box-Cox or Yeo-Johnson transformations
      - Exponential transformation for left-skewed data
   2. Distribution-specific models:
      - Using appropriate probability distributions (e.g., gamma, Poisson, negative binomial)
      - Generalized linear models with appropriate link functions
   3. Quantile regression to model different parts of the distribution
   4. Nonparametric methods that don't make distributional assumptions:
      - Kernel density estimation
      - Rank-based statistical methods

10. **Methods to reduce dimensionality but maintain critical data information within feature matrix?**

    * Linear methods: PCA, LDA
    * Nonlinear methods: t-SNE, UMAP
    * Feature selection methods: Lasso or Ridge (L1/L2 reg), tree based model feature importance

11. **How is the coefficient in logistic regression estimated by Maximum Likelihood Estimation, and how is the loss function derived?**

    Logistic regression coefficients are estimated through Maximum Likelihood Estimation (MLE), which finds the parameters that make the observed data most probable. 

    **Probability Model in Logistic Regression**

    In logistic regression, we model the probability of a binary outcome (y = 1) as:
    $$
    P(y = 1|x) = σ(β_0 + β_1x_1 + ... + β_px_p) = σ(β^Tx)
    $$
    Where:

    - $σ$ is the sigmoid function: $σ(z) = 1/(1 + e^{-z})$
    - $β$ represents the coefficient vector
    - $x$ represents the feature vector

    **Maximum Likelihood Estimation Process**

    1. **Likelihood Function Construction**: For a dataset with $n$ independent observations, the likelihood is:
       $$
       L(β) = \prod_{i=1}^n P(y = 1|x_i)^y_i \times (1 - P(y = 1|x_i))^{1-y_i}
       $$
       This gives the probability of observing our entire dataset given parameters $\beta$.

    2. **Log-Likelihood Transformation**: Taking the natural log converts the product to a sum, which is easier to work with:
       $$
       \ell (β) = \prod_{i=1}^n [y_i \log(P(y = 1|x_i)) + (1-y_i) \log(1 - P(y = 1|x_i))]
       $$
       Or, using the sigmoid function:
       $$
       \ell(\beta) = \prod_{i=1}^n [y_i \log(\sigma(\beta^T x)i) + (1-y_i) \log(1 - \sigma(\beta^T x_i))]
       $$

    3. **Maximizing the Log-Likelihood**: We want to find $\beta$ that maximizes $\ell(\beta)$, which is equivalent to minimizing $-\ell(\beta)$.

    **Loss Function Derivation**

    The loss function in logistic regression is the negative log-likelihood:
    $$
    Loss(\beta) = -\ell(\beta) = -\prod_{i=1}^n [y_i \log(\sigma(\beta^T x_i)) + (1-y_i) \log(1 - \sigma (\beta^T x_i))]
    $$
    This is the binary cross-entropy loss, which penalizes confident incorrect predictions more heavily than uncertain ones.

    **Finding the Optimal Coefficients**

    Since the loss function has no closed-form solution:

    1. We use numerical optimization techniques, usually gradient descent:
       - Calculate the gradient of the loss with respect to $\beta$
       - Update $\beta$ iteratively: $\beta_{new} = \beta_{old} - \alpha \times \nabla Loss(\beta)$
       - Continue until convergence (gradient becomes approximately zero)
    2. The gradient of the log-likelihood is: $\nabla \ell(\beta) = \prod_{i=1}^n x_i(y_i - \sigma(\beta^T x_i))$
    3. At convergence, the estimated coefficients $\hat{\beta}$ represent the maximum likelihood estimate.

    This process yields coefficients that best explain the observed relationship between features and outcomes in your dataset.

11. **How does decision tree classifier work mathmetically? How does a decision tree classifier decide on its split (entropy, purity, information gain)? What about decision tree regressor - how does split determined?** 

    

12. Is decision tree greedy or not? what are the pros and cons? What's the solution of the cons?

13. How Gini-Index is calculated in a DT classifier?

14. When evaluating model with imbalanced data, why precision and recall are better choice than ROC curve?

    ROC (TPR vs FPR)

    Recall = TP / (TP + FN); Precision = TP / (TP + FP)

    TPR = TP / (FN + TP); FPR = FP / (TN + FP)

15. Difference between boosting and bagging?

16. Random forest is a modified version of bagging algorithm, what is the one unique step that's different from bagging?

    Bagging has one problem "multi-colinearity". Random forest introduces stochasticity by bootstrapping features.

17. Why transformer is better than Recurrent Neural Network?

18. Why BoW, TFIDF are worse than word2vec, Glove, and BERT? What does semantic similarity mean - describe it in high dim space.

19. How does PCA work? How to calculate principle components mathmetically? What is the assumption of PCA and when it does not met? If assumption is not met, what are the alternatives?

20. Describe how would you plot ROC curve from scratch without using python package?

21. How do you choose different kernels when building a SVM model?

22. Best practices of pruning a decision tree?

23. What are the differences between Adaboost and Gradient Boosting?

24. Given 2 observations and 5 features, how do you calculate the euclidean distance?

25. How does (L2) regularization work in Gradient Boosting mathematically?

26. How does k-means work step by step? How to find the best k? If you run k-means multiple times, do you expect every clustering result is the same and why?

    Elbow plot and Silhouette plot.

27. Definition of conditional probability and Bayes Theorem?

28. How does kNN imputation work for imputing missing values step by step?

29. **Is the loss function of Neural Network convex or not and why? if not convex, what are the consequences? What's the solution for that? Why SGD can also reduce overfitting as well?**

    **1. Convexity of the Neural Network Loss Function**  

    The loss function of a neural network is **generally non-convex** due to the following reasons:  

    - **Composition of Non-linear Activation Functions** (e.g., ReLU, sigmoid, tanh) introduces non-linearities, making the loss landscape highly non-convex.  
    - **Multiple Layers and Interactions** between weights lead to complex, non-convex optimization surfaces with many local minima and saddle points.  
    - **Non-Convexity of Deep Architectures** Even simple neural networks with one hidden layer can have non-convex loss functions if non-linear activations are used.  

    **Exception:** If the network is a simple linear model (no hidden layers, no non-linear activations), the loss (e.g., mean squared error) is convex.  

    **2. Consequences of Non-Convexity**  

    - **No Guarantee of Global Optimum:** Gradient-based methods (e.g., SGD) may converge to local minima or saddle points instead of the global minimum.  
    - **Sensitivity to Initialization:** Different initial weights can lead to different solutions.  
    - **Optimization Challenges:** The loss landscape may have flat regions (vanishing gradients) or sharp minima (poor generalization).  

    **3. Solutions for Non-Convex Optimization**  

    Despite non-convexity, neural networks can still be trained effectively due to:  
    - **Stochastic Gradient Descent (SGD) and Variants (Adam, RMSprop):**  
      - Noise in SGD helps escape poor local minima.  
      - Adaptive optimizers handle saddle points better.  
    - **Good Initialization (He, Xavier):** Helps start in a favorable region of the loss landscape.  
    - **Batch Normalization:** Smoothens the optimization landscape.  
    - **Skip Connections (ResNet):** Eases training by mitigating vanishing gradients.  
    - **Ensemble Methods:** Training multiple models reduces reliance on a single local minimum.  

    **4. Why SGD Can Reduce Overfitting**  

    SGD acts as an **implicit regularizer** due to:  
    - **Noise in Updates:** The stochasticity prevents the model from fitting the training data too precisely.  
    - **Preference for Flat Minima:** SGD tends to converge to wider minima, which generalize better than sharp minima (as per the "flat minima" hypothesis).  
    - **Early Stopping:** SGD allows training to be stopped early (before full convergence), preventing overfitting.  

    Thus, SGD not only optimizes the loss but also helps improve generalization.  

    **Summary**  

    - **Loss is non-convex** due to non-linearities and deep architectures.  
    - **Consequences:** Local optima, sensitivity to initialization, optimization difficulties.  
    - **Solutions:** SGD variants, good initialization, batch norm, skip connections.  
    - **SGD reduces overfitting** via noise, flat minima preference, and early stopping.  

30. ANN training loss is not decreasing, what are the possible reasons?

    * underfitting due to large regularization; large learning rate; improper initiation e.g. init weight is 0; vanishing gradient

31. What is the loss function of a Neural Network based multi-class classifier? why it is a good loss function?

    Categorical cross entropy (one-hot encoded classes); sparse categorical cross entropy (label / integer encoded classes)

32. What is the loss function of a XGBoost based multi-class classifier? why it's a good loss function? how is the mlogloss calculated in a binary classifer? what does the LOG penalize heavily on? why mlogloss is better than accuracy?

    mlogloss - multi-class log loss.

    Measure how well model predicts probabilities for each class; penalize overconfidence in incorrect predictions; align with softmax objective.

33. How does XGBoost enable multiclass classification? 

    1 tree per class per iteration; gradient descent; aggregation across trees - summing contribution from all trees for each class and applying softmax