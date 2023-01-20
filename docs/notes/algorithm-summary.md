# ML Algorithms Summary

* **What is hypothesis testing?**

  Hypothesis testing is a formal process through which we evaluate the validity of a statistical hypothesis by considering evidence for or against the hypothesis gathered by random sampling of the data.

  1. <u>State the hypotheses</u>, typically a null hypothesis, $H_0$ and an alternative hypothesis, $H_1$, that is the negation of the former.
  2. <u>Choose a type of analysis and test statistics</u>, i.e. how to use sample data to evaluate the null hypothesis. Typically, this involves choosing a single test statistic.
  3. <u>Sample</u> data and <u>compute the test statistic</u>.
  4. Use the value of the test statistic to either <u>reject or not reject</u> the null hypothesis.

* **What is stepwise variable selection and validation?**

  Selecting optimal subsets of predictors through:

  * Stepwise variable selection - iteratively building an optimal subset of predictors by optimizing a fixed model evaluation metric each time.
  * Selecting an optimal model by evaluating each model on validation set.

  In <u>forward selection</u>, we find an 'optimal' set of predictors by iterative building up our set.

  1. Start with the empty set $P_0$, construct the null model $M_0$.

  2. For $k=1, ..., J$

     2.1 Let $M_{k-1}$ be the model constructed from the best set of $k-1$ predictors $P_{k-1}$.

     2.2 Select the predictor $X_{n_k}$ not in $P_{k-1}$, so that the model constructed from $P_k = X_{n_k} \cup P_{k-1}$ optimizes a fixed metric (can be p-value, F-stat, validation MSE, R-square, or AIC/BIC on training set)

     2.3 Let $M_k$ denote the model constructed from the optimal $P_k$.

  3. Select the model $M$ amongst $\{M_0, M_1, ...M_j\}$ that optimizes the fixed metric.

* **How to perform a k-fold cross validation?**

  Given a dataset $\{X_1, ..., X_n\}$, where each $\{X_1, ..., X_n\}$ contains $J$ features.

  To ensure that every observation in the dataset is included in at least one training set and at least one validation set, we use the K-fold validation:

  * Split the data into K uniformly sized chunks. $\{C_1, ..., C_K\}$
  * We create K number of training/ validation splits, using one of the K chunks for validation and the rest for training.

  We fit the model on each $K-1$ training set, denoted $\widehat{f}_{C_{-i}}$, and evaluate it on the one corresponding validation set $\widehat{f}_{C_{-i}}(C_i)$. The cross validation is the performance of the model averaged across all validation sets:

  $CV(model) = {1\over K}\sum^K_{i=1} L(\widehat{f}_{C_{-i}}(C_i))$.

* **What is the k-Nearest Neighbors algorithm?**

  kNN is a very human way of decision making by similar examples. It is a non-parametric learning algorithm.

  The kNN algorithm is as follows:

  Given a dataset $D = \{(x^{(1)},y^{(1)}), ..., (x^{(N)},y^{(N)})\}$. For  every new $x$:

  1. Find the $k$-number of observations in $D$ most similar to $x$. They are the $k$-nearest neighbors of $x$.

  2. Average the output of the $k$-nearest neighbors of $x$.

     $\widehat{y} = {1\over K} \sum^K_{k=1} y^{(n_k)}$.

  Note that in building a kNN model for prediction (non-parametric), we did not compute a <u>closed form</u> for $\widehat{f}$.

* **What is the math behind PCA? (Simple)**

  Let $Z$ be the $n \times p$ matrix consisting of columns $Z_1,..., Z_p$ (The resulting PCA vectors), $X$ be the $n \times p$ matrix of $X_1, ..., X_p$ of the original data variables (each standardized to have mean 0 and variance 1, and without the intercept), and let $W$ be the $p \times p$ matrix whose columns are the eigenvectors of the square matrix $X^T X$, then

  $Z_{n \times p} = X_{n\times p} W_{p \times p}$

  To implement PCA, perform the following steps:

  1. Standardize each of your predictors (so they each have mean = 0, variance =1).
  2. Calculate the eigenvectors of the $X^T X$ matrix and create the matrix with those columns, $W$, in order from largest to smallest eigenvalue.
  3. Use matrix multiplication to determine $Z=XW$.

* **What is the math behind PCA ? (Complex)**

  1. Consider the <u>data matrix</u> $X \in R^{n \times p}$. We center the predictors by subtracting the sample mean. Then the <u>centered data matrix</u> is $\stackrel{\sim}{X} = (\stackrel{\rightarrow}{x}_1 - \widehat{\mu}_1, ..., \stackrel{\rightarrow}{x}_p - \widehat{\mu}_p)$. 

  2. Consider the <u>Covariance Matrix</u> $S = {1\over n-1} \stackrel{\sim}{X}^T \stackrel{\sim}{X}$. It is symmetric, so it permits an orthonormal eigen-basis ($V$) and eigen-decomposition.

  3. <u>Compute eigenvectors and eigenvalues</u> from the covariance matrix $S$.

     $Sv_i = \lambda_i v_i$

     $S = V\Lambda V^T$

  4. <u>Sort</u> the eigenvectors $v_i$ by decreasing eigenvalues.

     The eigenvalues can be sorted in $\Lambda$ as $\lambda_1 > \lambda_2 > ... > \lambda_p$. The eigenvector $v_i$ is called the $i$th principal component of $S$.

  5. <u>Choose</u> $k$ eigenvectors with the largest eigenvalues to from a $d\times k$ dimensional matrix $W$. 

  6. <u>Transform</u> the samples onto the new subspace.

     $y_{n \times k} = x W$.

* **How to perform iterative PCA for imputation?**

  We want our imputations to take into account:

  1. <u>Relationship between variables</u> (we want to impute missing values in one predictor using values from the other(s)).
  2. Global <u>similarity between individual observations</u> (two observations might have similar values for most of the predictors). 

  <u>Iterative PCA for imputation</u> is used for missing at random scenario:

  1. <u>Initialize</u> imputation with reasonable value (e.g. mean)
  2. <u>Iterate</u> until convergence:
     1. Preform <u>PCA</u> on the complete data
     2. Retain first <u>M components</u> of PCA (in example M=1)
     3. <u>Project</u> imputation into PCA space
     4. <u>Update</u> imputation using projection value

  The choice of the number of components to use is made by cross-validation.

  In practice the issue of this method is it can overfit, especially in a sparse matrix with many missing values. So often a regularized PCA approach is used which shrinks the singular values of the PCA, making updates less aggressive.

* **How do we find the optimal solution of a loss function?**

  <u>Gradient descent:</u> 

  Initialize a random point, repeatedly update it based on the gradient of loss and a learning rate lambda. 

  Initialize $w_0$ randomly, for t = 1:T (iterations), update $w$ by rule: $w_{t+1} = w_{t}-\lambda_t \nabla_w L(w_t)$. 

  Learning rate $\lambda$: if it's too large, the updates are unstable and can diverge. If it's too small, the updates are stable but very slow.

  <u>Gradient descent with Momentum:</u>

  Based on gradient descent algorithm, it introduces a momentum coefficient $\alpha \in [0,1)$ so that the updates have 'memory'. 

  Momentum is a nice trick that can help speed up convergence. Generally we choose α between 0.8 and 0.95, but this is problem dependent. 

  Initialize $w_0$ randomly, initialize $\delta_0$ to the zero vector, for t=1:T (iterations), update: $\delta_{t+1} = -\lambda \nabla_{w_{t}}L(w_{t}) + \alpha \delta_{t}\\ w_{t+1} = w_{t} + \delta_{t+1}$. 

  <u>Stochastic gradient descent:</u> 

  Considering using all the data is computational expensive, SGD randomly picks a subsample (even one datum works) and compute gradient. 

  Mini-batch gradient descent: Randomly shuffle examples in the training set and compute the average gradient among samples. People commonly use the term SGD for mini-batch optimization.

  The drawback is it's too noisy. The merit is it's very fast. This method produces an unbiased estimator of the true gradient. This is the basis of optimizing ML algorithms with huge datasets.

  Computing gradients using the full dataset is called batch learning, using subsets of data is called mini-batch learning.

* **How to learn a decision tree? (Simple)**

  Learning the smallest optimal decision tree for any given dataset is <u>NP complete</u> (intractable) for numerous simple definitions of optimality. Instead, we will seek a reasonable model using a <u>greedy</u> algorithm.

  1. Start with an empty decision tree (undivided feature space)
  2. Choose the optimal predictor on which to split and choose the <u>optimal threshold</u> value for splitting
  3. Recurse on each new node until <u>stopping condition</u> is met.

* **How to build a Decision Tree specifically? (Complex)**

  A decision tree is built top-down from a root node and involved partitioning the data into subsets that contain instances with similar values.

  <u>Steps to build a DT</u>

  1. Start with the original dataset as the root node. Calculate entropy of the target $H(Y) = -\sum_{i=1}^c p(Y=i)\log_2 p(i)$.

  2. Calculate conditional entropy for the target and each feature (measure the uncertainty associated with target given each feature): $H(Y|X1),H(Y|X2)$ (Note that they should be smaller than $H(Y)$)

  3. Calculate information gain for each feature

     $IG(Y,X1)=H(Y)−H(Y|X1)\\IG(Y,X2)=H(Y)−H(Y|X2)$

  4. Choose the feature with the largest IG as the decision node.

     A branch with entropy of 0 is a leaf node. A branch with entropy more than 0 needs further splitting.

  5. Recurse on non-leaf branches until all data is classified.

* **How to prune a complex tree?**

  A common pruning method is <u>cost complexity pruning</u>, where by we select from an array of smaller subtrees of the full model that optimizes a balance of performance and efficiency.

  That is, we measure $C(T) = Error(T) + \alpha|T|$.

  where $T$ is a decision tree, $|T|$ is the number of leaves in the tree and $\alpha$ is the parameter for penalizing model complexity.

  <u>The pruning algorithm</u>:

  1. Start with a full tree $T_0$ (Each leaf node is pure)

  2. Replace a subtree in $T_0$ with a leaf node to obtain a pruned tree $T_1$​. We want the subtree to minimize

     ${Error(T_0) - Error(T_1)\over |T_0|-|T_1|}$

  3. Iterate this pruning process to obtain $T_2, ...,T_L$.

  4. Select the optimal tree $T_i$ by cross validation. 

  We are explicitly optimizing cost complexity $C(T)$ at each step.

* **What is bagging? ** 

  <u>Bootstrap aggregation</u>, or bagging, is a popular ensemble method that fits a decision tree on different bootstrap samples of the training dataset, and combines predictions from multiple-decision trees through a <u>majority voting</u> mechanism.

  The key idea is: One way to adjust for the <u>high variance</u> of the output of an experiment is to perform the experiment multiple times and then average the results.

  1. <u>Bootstrap</u>: we generate multiple samples of training data, via bootstrapping. It involves selecting examples randomly with replacement. We train a full decision tree on each sample of data.
  2. <u>AGgregatiING</u>: for a given input, we output the averaged outputs of all the models for that input.

  Steps:

  1. Multiple subsets are created from the original dataset by bootstrapping, selecting observations with replacement.
  2. A base model (weak model) is created on each of these subsets.
  3. The models run in parallel and are independent of each other.
  4. The final predictions are determined by combining the predictions from all the models.

* **How to measure the feature importance in Random Forest?**

  1. <u>Mean Decrease in Impurity (MDI)</u>

     Calcualte the total amount that the MSE or RSS (for regression) or Gini index or entropy (for classification) is decreased due to splits over a given predictor, averaged over all trees.

     Basically, if a feature can do better in variance/ error reduction or impurity reduction, on average, then it's a more important feature.

     It is implemented in `feature_importances_` in scikit-learn Random Forest.

     <u>Advantage</u>: computationally efficient.

     <u>Disadvantage</u>: 

     * this method tends to prefer (select as important) numerical features and categorical features with high cardinality. 
     * For correlated features, it can select one of the feature and neglect the importance of the other.

  2. <u>Permutation based Feature Importance</u>

     1. Record the prediction accuracy on the out-of-bag (oob) samples for each tree.

     2. Randomly shuffles each feature and compute the change in the model’s performance. 
     3. The decrease in accuracy as a result of this permuting is averaged over all trees. This is used as a measure of the importance of variable. 

     That is, the features which impact the performance the most are the most important one.

     It is implemented in `permutation_importantce` in scikit-learn. 

     <u>Disadvantages</u>: 

     * computationally expensive. 
     * Have problem with highly-correlated features, it can ignore their importance.

* **What is Boosting?**

  Boosting is a sequential process, where each subsequent model attempts to correct the errors of the previous model. The succeeding models are dependent on the previous model. In this technique, learners are learned sequentially with early learners fitting simple models to the data and then analyzing data for errors. In other words, we fit consecutive trees (random sample) and at every step, the goal is to solve for net error from the prior tree. When an input is misclassified by a hypothesis, its weight is increased so that next hypothesis is more likely to classify it correctly. By combining the whole set at the end converts weak learners into better performing model.

  It trains a large number of "weak" learners <u>in sequence</u>. A weak learner is a constrained model (limit the max depth of each decision tree). Each one in the sequence focuses on <u>learning from the mistakes of the one before it</u>. By <u>more heavily weighting in the mistakes in the next tree</u> (the correct predictions are identified and less heavily weighted in the next tree), our next tree will learn from the mistakes. Boosting additively combines all the weak learners into a single strong learner, and gets a boosted tree. The ensemble is a <u>linear combination</u> of the simple trees, and is more expressive.

  Steps:

  1. A subset is created from the original dataset. Initially, all data points are given equal weights. A base model is created on this subset. This model is used to make predictions on the whole dataset.
  2. Errors are calculated using the actual values and predicted values. The observations which are incorrectly predicted, are given higher weights. Another model is created and predictions are made on the dataset. This model tries to correct the errors from the previous model. 
  3. Similarly, multiple models are created, each correcting the errors of the previous model. The final model (strong learner) is the weighted mean of all the models (weak learners).

* **What is Gradient Boosting?** [Resource 1](https://towardsdatascience.com/all-you-need-to-know-about-gradient-boosting-algorithm-part-2-classification-d3ed8f56541e) [Resource 2](https://towardsdatascience.com/all-you-need-to-know-about-gradient-boosting-algorithm-part-1-regression-2520a34a502)

  Gradient boosting looks at the difference between its current approximation, and the known correct target vector, which is called the <u>residual</u>. It fits the next model to the residuals of the current model.

  <u>Gradient boosting algorithm</u> (simple version):

  1. Fit a simple model $T^{(0)}$ on the training data $\{(x_1, y_1), ..., (x_N, y_N)\}$. 

  2. Set $T \leftarrow T^{(0)}$. Compute the residuals $\{r_1, ..., r_N\}$ for $T$.

  3. Fit a simple model $T^{(1)}$, to the current residuals, i.e. train using $\{(x_1, r_1), ..., (x_N, r_N)\}$.

  4. Set $T \leftarrow T+ \lambda T^{(1)}$, where $\lambda$ is a learning rate. 

     * Specifically for <u>classification</u> we do: $T \leftarrow T + \lambda \cdot \gamma$. $\gamma$ is the value we want to add to the previous prediction. It is computed for each terminal node, as $\lambda = {\sum_{x_i \in R_j} (y_i - p)\over \sum_{x_i \in R_j} p(1-p)}$ where $p$ is the predicted probability. 

       Compute residuals using the new $T$. The new residuals will be $r_n \leftarrow r_n - \lambda \cdot \gamma$.

     * Specifically for <u>regression</u> we do: $T \leftarrow T + \lambda \cdot \gamma$. $\gamma$ is the predicted value for each terminal node. $\lambda$ is the learning rate between 0 and 1.

       Compute residuals again using the new $T$.

  5. Repeat previous steps until stopping condition is met.

  Note that the residuals can be compute by the difference between the true values and the predicted values, or the gradients of the loss function.

* **What is AdaBoost?**

  Adaptive Boosting (AdaBoost) is a gradient boosting method, often used for classification. In AdaBoost, we are not using Error function (which is not differentiable), instead we use a differentiable function as a good indicator of classification error: <u>exponential loss</u>.

  $ExpLoss = {1\over N}\sum^N_{n=1} \exp(-y_n \widehat{y}_n), y_n \in \{-1,1\}$

  We can compute the gradient for ExpLoss as the residuals we want the next model to be trained on. 

  That is, gradient descent with exponential loss means iteratively training simple models that focuses on the points misclassified by the previous model. 

  <u>AdaBoost Algorithm</u>:

  1. All data points will be assigned some weights. Initially, all the weights are equal. $w(x_i, y_i) = 1/N$.

  2. At the $i$th step, fit a simple classifier $T^{(i)}$ on weighted training data $\{(x_1, w_1y_1),...,(x_N, w_Ny_N)\}$.

  3. Update the weights: 

     $w_n \leftarrow {w_n \exp(-\lambda^{(i)} y_n T^{(i)}(x_n))\over Z}$ where $Z$ is the normalizing constant for the collection of updated weights

  4. Update $T \leftarrow T+ \lambda^{(i)} T^{(i)}$.

  Unlike in the case of gradient boosting for regression, we can analytically solve for the optimal learning rate for AdaBoost, by optimizing the ExpLoss for the $\lambda$. Doing so we get $\lambda ^i = 1/2 \log ({ 1- \epsilon\over \epsilon})$, $\epsilon = \sum^N_{n=1} w_n I(y_n\neq T^{(i)}(x_n))$.

* **What is the k-means algorithm?**
  1. <u>Randomly assign</u> each observation to one of K clusters at random.
  2. Repeat the following two steps until clusters do not change:
     * For each cluster k, compute the <u>cluster centroid</u> $\overline{x_k}$ (<u>The variable-wise average of the observations in cluster k</u>).
     * Given the K centroids, <u>reassign all observations to clusters based on their closeness to the centroids</u>.

* **What are steps of performing bayesian analysis?**

  1. Formulate a probability model for the data.

     For example, if the outcome is whether the user clicked the button or not, we are going to form a Bernoulli model. 

  2. Decide on a prior distribution for the unknown model parameters.

     (Note: usually improper prior distribution lead to proper posterior distributions, but one needs to check. A proper prior distribution always leads to a proper posterior distribution.)

  3. Observe the data, and construct the likelihood function based on the data.

     (Note: the likelihood function is the joint probability of the data.)

  4. Determine the posterior distribution 

     (Note: Posterior $\propto$ Prior $\times$ Likelihood , obtained by bayes rule. Usually the normalizing constant cannot be determined analytically.)

  5. Summarize important features of the posterior distribution, or calculate quantities of interest based on the posterior distribution.

     * Posterior mean

     * Posterior mode

     * 95% central posterior interval for $\theta$

       Narrower than frequentist 95% confidence interval.

     * Highest posterior density (HPD) region for $\theta$- shortest interval with specified probability. Usually more difficult to compute than central intervals. 

* **How to implement an MCMC sampler to obtain simulated parameter values and do numerical summaries?**

  To obtain simulated parameter values, do:

  1. Run several parallel MCMC samplers with different starting values (preferably widely dispersed)
  2. Simulate values from the Markov Chains for a <u>'burn in' period</u> (before the Markov Chains have converged to the stationary distribution), and discard the burn-in simulations.
  3. Save simulated values after burn-in period. These will be the simulated values on which to perform inferential summaries.

  Things need to take care of without packages:

  * Pick a good proposal distribution $\tilde{p}(\Delta)$ for RWM where the jumps are not too big or too small.
  * Simulate from the probability distributions
  * Write code to perform Markov Chain simulation, check for convergence, etc.

  With packages: JAGS, pymc3, etc, we need to

  * specify the model for data
  * specify logistical issues concerning MCMC simulation (starting values, burn-in period, iterations to run after burn-in, etc)

  To summarize model results with the simulated values across all chains after the burn-in iterations:

  * Approximate posterior mean and standard deviations of parameters by their sample counterparts.
  * Can compute 95% central posterior intervals from the 2.5% and 97.5% of the sample of simulated values.