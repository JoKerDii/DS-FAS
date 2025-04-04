# Machine Learning Concepts Q&A (I)

**What is ML workflow?**

1. Should I use ML on this problem? 
   * Is there a pattern to detect? 
   * Can I solve it analytically? 
   * Do I have data? 
2. Gather and organize data. 
3. Preprocessing, cleaning, visualizing. 
4. Establishing a baseline. 
5. Choosing a model, loss, regularization, ... 
6. Optimization (could be simple, could be a Phd...). 
7. Hyperparameter search. 

## 1. Regression

* **We have had a train-test split. Why we need a validation set?** 

  The *training set* is the data we have seen. We train different types of models with various different hyperparameters and regularization parameters on this data.

  The *validation set* is used to <u>compare different models</u>. We use this set to <u>tune our hyper-parameters</u> i.e. find the optimal set of hyper-parameters and pick the best model.

  The *test set* is used to report the scores using our best model. <u>We don't tune parameters using test set</u>, because we want to know the generalizability of our model, i.e. how our model performs on a new data set.

* **For which models do the unit / scale of the predictors matter?**

  <u>k-NN</u>. Scaling affects distance metric - <u>Euclidean distance measure</u>, which determines what '"neighbor" means. All such distance based algorithms are affected by the scale / magnitude of the variables.

  <u>Clustering</u>. It is important to standardize variables before running Cluster Analysis. It is because <u>cluster analysis techniques depend on the concept of measuring the distance between the different observations we're trying to cluster</u>. If a variable is measured at a higher scale than the other variables, then whatever measure we use will be overly influenced by that variable.

  <u>Principal Component Analysis (PCA)</u>. Prior to PCA, it is critical to standardize variables. It is because <u>PCA gives more weight to those variables that have higher variances</u> than to those variables that have very low variances. In effect the results of the analysis will depend on what units of measurement are used to measure each variable. Standardizing raw values makes equal variance so high weight is not assigned to variables having higher variances.

  <u>Support Vector Machine (SVM)</u>. <u>SVM model is optimized by maximizing the distance between hyperplane and the support vectors</u> so it is required to scale variables prior to training a SVM model.

  <u>Lasso and Ridge Regression</u>. <u>They put constraints on the size of the coefficients associated to each variable</u>. However, this value will depend on the magnitude of each variable.

  Scaling does not affect statistical inference in simple regression models, as the estimates are adjusted appropriately and the p-values will be the same. But it does <u>removes collinearity</u>. Interaction term would introduced collinearity. Scaling the one of the two variables can remove collinearity.

* **Compare normalization and standardization?**

  The point of normalization and standardization is to make the variables take on a more manageable scale. Ideally, we should standardize or normalize all the variables at the same time. But they are not always needed.

  Normalization makes data go from 0 to 1. It is widely used in image processing and computer vision, where pixel intensities are non-negative and are typically scaled from a 0-255 scale to a 0-1 range for a lot of different algorithms.

  Normalization is very useful in neural networks as it leads to the algorithms converging faster.

  Normalization is useful when the data <u>does not have a discernible distribution</u> and we are not making assumptions about the data's distribution.

  <u>Standardization maintains outliers</u> whereas normalization makes outliers less obvious. Standardization is preferred when the outliers are useful.

  Standardization is useful when we assume the data come from a <u>Gaussian distribution</u> (approximately).

  Standardization makes it realistic to interpret the intercept term as the expected value of $Y_i$ when the predictor values are set to their means.

* **What is Bootstrapping?**

  Bootstrapping is a procedure for resampling a dataset <u>with replacement</u> to produce an <u>empirical distribution</u> of the value of interest.

  Bootstrap is useful because it's easy to apply and require no assumption regarding the distribution of the data.

  The bootstrap method can be used to estimate a quantity of a population. This is done by repeatedly taking small samples with replacement, calculating the statistic, and taking the average of the calculated statistics.

* **What is confidence interval and how to interpret it?**

  A confidence interval is a range of values that is likely to include a parameter of interest with some degree of certainty or "confidence."

  If we were to compute 95% confidence intervals for each of K repeated samples, we would expect <u>0.95*K of those confidence intervals to contain the true parameter of interest</u>.

* **What is multicollinearity?**
  
  <u>Overview</u>: 
  
  Multicollinearity occurs when independent variables in a regression model are correlated. This correlation is a problem because independent variables should be independent. If the degree of correlation between variables is high enough, it can cause problems when you fit the model and interpret the results.
  
  <u>Why multicolinearity should not exist?</u>
  
  There is an assumption when we do regression analysis. That is, the independent variables should not be correlated among themselves. Because regression is going to capture the effect of independent variable on the dependent variable by regression coefficients. When we interpret coefficients we say, the average change of dependent variable is beta, for 1 unit change in the independent variable X, keeping all other variables constant. 
  
  <u>What problem does multicollinearity cause?</u>
  
  Multicollinearity is a problem because <u>it generates high variance of the estimated coefficients and so diminishes the statistical significance of an independent variable</u>. If two independent variables are correlated, it is hard to distinguish the effect of these two variables on the dependent variable. Other things being equal, the larger the standard error of a regression coefficient, the less likely it is that this coefficient will be statistically significant. 
  
  <u>Possible way to detect multicollinearity?</u>
  
  It is possible that the adjusted R squared for a model is pretty good and even the overall F-test statistic is also significant but some of the individual coefficients are statistically insignificant. This is a possible indication of the presence of multicollinearity since multicollinearity affects the coefficients and corresponding p-values, but it does not affect the goodness-of-fit statistics or the overall model significance.
  
  <u>How to measure multicollinearity?</u>
  
  A very simple test known as the VIF test is used to assess multicollinearity in our regression model. The variance inflation factor (VIF) identifies the strength of correlation among the predictors.
  
  VIF tells us the factor by which the correlations amongst the predictors inflate the variance. VIFs do not have any upper limit. The lower the value the better. VIFs between 1 and 5 suggest that the correlation is not severe enough to warrant corrective measures. VIFs greater than 5 represent critical levels of multicollinearity where the coefficient estimates may not be trusted and the statistical significance is questionable.
  
* **What does regularization help with?**

  We have some pretty <u>large and extreme coefficient values with high variance</u> in our models. We can clearly see some overfitting to the training set. In order to reduce the coefficients of our parameters, we can introduce a penalty term that penalizes some of these extreme coefficient values.

  1. <u>Avoid overfitting</u>. Reduce features that have weak predictive power. (Idea: reduce variance by increase bias)
  2. Discourage the use of a model that is too complex.

* **Compare Ridge and Lasso regression?**

  Ridge and Lasso regressions are linear regression with different regularization - L2 and L1 regularization. 

  Ridge regression is penalized for the sum of squared values of the coefficients. The penalty term is the product of a parameter $\lambda$ and L2-norm of the coefficient.

  Lasso regression is penalized for the sum of absolute values of the coefficients. The penalty term is the product of a parameter $\lambda$ and L1-norm of the coefficient.

  Differences:
  
  1. Since Lasso regression tend to produce zero estimates for a number of model parameters - we say that <u>Lasso solutions are sparse</u> - we consider to be a method for <u>variable selection</u>.
  2. In Ridge Regression, the penalty term is proportional to the L2-norm of the coefficients whereas in Lasso Regression, the penalty term is proportional to the L1-norm of the coefficients.
  3. <u>Ridge Regression has a closed form solution</u>, <u>while Lasso Regression does not</u>. We often have to solve this iteratively. In the sklearn package for Lasso regression, there is a parameter called `max_iter` that determines how many iterations we perform.

## 2. PCA

* **What is PCA?**

  PCA is a framework for <u>dimensionality reduction</u>. Considering a new system of coordinates, namely a new set of predictors, each of which is a <u>linear combination</u> of the original predictors, that captures the <u>maximum amount of variance</u> in the observed data. The basic assumption in PCA is that higher variance indicates more importance.

  The new set of predictors are called "<u>principal components</u>". The principal components consist an <u>m-dimensional orthonormal system of coordinates</u>. PCA sorts the axis such that the largest variance goes along the first predictor, the second largest variance goes to the second predictor, and so on. Then we can build a linear regression model using the new predictors.

  To reduce the dimension, we can keep just a few of the PCs and drop the rest.

* **Why (not) PCA?**

  PCA is great for

  1. <u>Speeding up</u> the training of a model without significantly decreasing the predictive ability relative to a model with all p predictors.
  2. <u>Visualizing</u> how predictive your features can be of your response, especially in the case of classification.
  3. <u>Reducing multicollinearity</u>, and thus potentially improving the computational time required to fit models.
  4. <u>Reducing dimensionality</u> in very high dimensional settings.

  PCA is not so good when

  1. <u>Interpretation</u> of coefficients in PCA is completely lost. So do not do PCA if interpretation is important.
  2. When the predictors' distribution deviates significantly from a <u>multivariable Normal</u> distribution.
  3. <u>When the high variance does not indicate high importance</u>.
  4. When the hidden dimensions are <u>not orthonormal</u>.

* **Assumption of PCA?**

  1. <u>Linear change of basis</u>: PCA is a linear transformation from a Euclidean basis (defined by the original predictors) to an abstract <u>orthonormal basis</u>. Hence, PCA assumes that such a linear change of basis is sufficient for identifying degrees of freedom and conducting dimensionality reduction.

  2. <u>Mean/variance are sufficient (data are approximately multi-variate gaussian):</u> In applying PCA to our data, we are only using the <u>means</u> (for standardizing) and the <u>covariance matrix</u> that are associated with our predictors. Thus, PCA assumes that such statistics are sufficient for describing the distributions of the predictor variables. This is true only if the predictors are drawn jointly from a <u>multivariable Normal distribution</u>. When the predictor distributions heavily violate this assumption, PCA components may not be as informative.

  3. <u>High variance indicates high importance:</u> This fundamental assumption is intuitively reasonable, since components corresponding to low variability likely say little about the data, but this is not always true.

  4. <u>Principal components are orthogonal</u>: PCA explicitly assumes that *intrinsic (abstract) dimensions* are orthogonal, which may not be true. However, this allowes us to use techniques from linear algebra such as the spectral decomposition and thereby, simplify our calculations.

## 3. Classification

* **What is generalization?**

  Generalization is model ability to predict held out data. Simple model cannot model data well. Complex model models also noise. 

* **How to evaluate how good my classifier is?**

  Metrics. 

  * Metrics on a dataset is the model performance that we care about. 
  * We typically cannot directly optimize for the metrics. 
  * Our loss function should reflect the problem we are solving. We then hope it will yield models that will do well on our dataset.

  Examples of metrics:

  * Accuracy

  * Recall: R = TP / all groundtruth instances

  * Precision: P = TP / all positive predictions

  * F1 score: Harmonic mean of precision and recall: 2 P*R/(P+R)

  * Precision-Recall curve: Trade-off between recall and precision using the decision threshold

  * Average Precision (AP): area under the PR curve

  * Receiver Operator Characteristic (ROC): Trade-off between false-positive-rate (FPR) and true-positive-rate (TPR) using the decision threshold

    Better in ROC ⇒ better in PR (not always vice-versa)

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

* **What the different types of cross-validation?**

  <u>Leave-p-out cross-validation:</u>

  We use p observations as the validation set and the remaining observations as the training set.

  <u>K-fold cross-validation:</u>

  The training set is randomly partitioned into k equal size subsamples. Of the k subsamples, a single subsample is retained as the validation data for testing the model, and the remaining k − 1 subsamples are used as training data. The cross-validation process is then repeated k times (the folds). The k results from the folds can then be averaged (or otherwise combined) to produce a single estimate.

* **Pros and Cons of Logistic Regression?**

  <u>Pros:</u>

  Quick to train, convex loss. Cross-entropy objective function for logistic regression is convex.

  Good accuracy for many simple data sets

  Resistant to overfitting if #data >= 10 #features.

  Good interpretability. Can interpret model coefficients as indicators of feature importance.

  <u>Cons:</u>

  Linear decision boundary is too simple for more complex problems.

* **Why do we care about convexity?** 

  Any local minimum is a global minimum.

  This makes optimization a lot easier because we don't have to worry about getting stuck in a local minimum。

* **What's the difference between linear classification and non-linear classification?**

  Linear classification means that the decision boundary is linear. 

* **What's the difference between parametric and non-parametric models?**

  Parametric models have fixed number of parameters.

  Non-parametric models have no fixed number of parameters. Parameters grow with the number of training data points. 

* **What is K Nearest Neighbors method?**

  The value of the target function for a new point is estimated from the known values of the nearest training examples. The distance is defined to be Euclidean. 

  NN algorithm does not explicitly compute decision boundaries but these can be inferred. The decision boundaries can be visualized by Voronoi diagram. It natually forms complex decision boundaries.

  <u>How to choose K?</u>

  Large K may leads to better performance / generalization. But if K is too large, we may end up looking at samples that are far away from the point. 

  We can use cross validation to find K. Rule of thumb is K< sqrt(n), where n is the number of training examples.

  <u>Issues and remedies:</u> 

  * Attributes with larger ranges are treated as more important. So we need to normalize or standardize the scale.
  * Irrelevant, correlated attributes add noise to distance measure. So we need to eliminate some attributes
  * High dimensional data increase computational costs. So we need to increase training data exponentially with dimension.
  * Expensive at run time.

* **How to make prediction using KNN?**

  For K=1, we predict the same value / class as the nearest instance in the training set.

  For K>1, find the K closest training examples, and either predict class by majority vote (in classification), or predict value by average weighted inverse distance (in regression).

  Ties may occur in a classification problem when K > 1. For binary classification, we choose an odd K to avoid ties. For multi-class classification, we decrease the value of K until the tie is broken. If that does not work, use the class by a 1NN classifier.

  Larger K: predictions have higher bias (Less true). Under fitting. Model is too simple.

  Smaller K: predictions have higher variance (Less stable / robust). Over fitting. Model is too complex.

* **What is the benefits of ROC curve and AUC score?**

  The ROC curve shows the performance of a classification model at all classification threshold. It allows us to find the classification threshold that gives the best FPR and TPR trade-off. 

  By summarizing the information in the ROC curve, we can compare our classifier against a perfect classifier and a random classifier. 

  We can compare models by comparing AUC score, which is the area under the ROC curve.

* **What is decision tree? Why decision tree?** 

  Decision tree is a graphical representation of making a decision based on certain conditions.

  <u>Decision tree properties:</u>

  * Choose an attribute on which to descend at each level
  * Condition on earlier choices
  * Restrict only one dimension at a time
  * Declare an output value when you get to the bottom
  * Not necessarily split each dimension once.

  It's like simple flow charts that can be formulated as mathematical models for classification and these models have the properties we desire:

  - Interpretable by humans
  - Have sufficiently complex decision boundaries
  - The decision boundaries are locally linear, each component of the decision boundary is simple to describe mathematically. 

* **Difference of decision tree on classification and regression?**

  Suppose each path from the room to a leaf is a region R of input space. Let a set of points be the training examples that fall into R.

  For classification, we return the <u>majority</u> class in the points of each leaf node.

  For regression we return the <u>average</u> of the outputs for the points in each leaf node.

* **How does a Decision Tree decide on its splits (what is the criteria for a split point)?**

  It finds the feature that best splits the target class into the purest possible children nodes. Eventually, decision tree aims to achieve the minimal average classification error.

  The measure of purity is called the <u>information</u>. It represents the <u>expected amount of information that would be needed to specify whether a new instance should be classified 0 or 1, given the example that reached the node</u>.

  <u>Entropy</u> on the other hand is a measure of impurity. The formula is $-p(a) * \log (p(a)) -p(b) * \log (p(b)) $. High entropy means being less predictable. Low entropy means being more predictable.

  By comparing the entropy before and after the split, we obtain the <u>information gain</u>, it is how much the information gained by doing the split using that particular feature. So the information gain can be calculate by entropy before the split minus the entropy after the split. If the IG=0, then the split is completely uninformative. If the IG is entropy before the split, then the split is completely informative.

  <u>Gini index</u> is another commonly used measure of purity. The formula of gini index is 1 minus the <u>squared probability</u> of each class. The optimal feature to split is chosen by minimizing the <u>weighted sum</u> of Gini index in each child node.

* **How to build a Decision Tree specifically?**

  A decision tree is built top-down from a root node and involved partitioning the data into subsets that contain instances with similar values.

  Steps to build a DT

  1. Calculate entropy of the target $H(Y) = -\sum_{i=1}^c p(Y=i)\log_2 p(i)$.

  2. Calculate conditional entropy for the target and each feature (measure the uncertainty associated with target given each feature): $H(Y|X_1),H(Y|X_2)$ (Note that they should be smaller than $H(Y)$)

  3. Calculate information gain for each feature

     $IG(Y,X_1) = H(Y) -H(Y|X_1)\\IG(Y,X_2) = H(Y) -H(Y|X_2)$

  4. Choose attribute with the largest IG as the decision node.

     A branch with entropy of 0 is a leaf node. A branch with entropy more than 0 needs further splitting.

  5. Recurse on non-leaf branches until all data is classified.

* **What is "stopping condition" in a decision tree?**

  To avoid overfitting, we could (limit the size of the tree)

  - Stop the algorithm at a particular depth. (=<u>not too deep</u>)
  - Don't split a region if all instances in the region belong to the same class. (=<u>stop when subtree is pure</u>)
  - Don't split a region if the number of instances in the sub-region will fall below pre-defined threshold (min_samples_leaf). (=<u>not too specific/small subtree</u>)
  - Don't use too many splits in the tree (=<u>not too many splits / not too complex global tree</u>)
  - Be content with <100% accuracy training set...
  
* **Why is using FPR and TPR not as good as precision and recall for evaluating models with unbalanced data?**

  When dealing with unbalanced data, some metrics might not be able to correctly reflect model performance. 

  <u>Definition</u>:

  In ROC curve, we look at TPR (True Positive Rate) = # True Positives / # Positives = Recall = TP / (FN + TP), and FPR (False Positive Rate) = # False Positives / # Negatives = FP / (TN + FP). ROC curve consists of many TPR and FPR through various probability thresholds.

  Recall = # True Positives / # Positives = TP / (TP + FN)

  Precision = # True Positives / # Predicted Positives = TP / (TP + FP). 

  <u>Difference</u>:

  The main difference is between precision and FPR. Precision measures the probability of a sample classified as positive to actually be positive. FPR measures the proportion of false positives within the negative samples.

  <u>Comparison</u>:

  <u>If there are a large number of negative samples, precision is better.</u> If the number of negative samples is large, TN is large and the denominator of the FPR is large. So FPR would be small and the FP is hard to detect. However, precision is not affected by a large number of negatives, because it measures the number of TP out of the predicted positives. 

  <u>All in all, precision measures the probability of correct detection of positive values, while FPR and TPR (ROC) measure the ability to distinguish between the classes.</u>

## 4. Ensemble

* **A summary of Ensembles**
  
  * Ensembles combine classifiers to improve performance. 
  * Boosting 
    * Reduce bias. 
    * Increases variance (large ensemble can cause overfitting). 
    * Sequential. 
    * High dependency between ensemble elements. 
  * Bagging
    * Reduce variance (large ensemble can’t cause overfitting). 
    * Bias isn’t changed
    * Parallel. 
    * Minimizes correlation between ensemble elements.
  
* **What is bagging? Why (not) bagging?** 

  <u>Bootstrap aggregation</u>, or bagging, is a popular ensemble method that fits a decision tree on different bootstrap samples of the training dataset, and combines predictions from multiple-decision trees through a <u>majority voting</u> mechanism.

  The key idea is: One way to adjust for the <u>high variance</u> of the output of an experiment is to perform the experiment multiple times and then average the results.

  1. <u>Bootstrap</u>: we generate multiple samples of training data, via bootstrapping. It involves selecting examples randomly with replacement. We train a full decision tree on each sample of data.
  2. <u>AGgregatiING</u>: for a given input, we output the averaged outputs of all the models for that input.

  Steps:

  1. Multiple subsets are created from the original dataset by bootstrapping, selecting observations with replacement.
  2. A base model (weak model) is created on each of these subsets.
  3. The models run in parallel and are independent of each other.
  4. The final predictions are determined by combining the predictions from all the models.

  Merits:

  * <u>High expressiveness</u>: by using larger trees it is able to approximate complex functions and decision boundaries.
  * <u>Low variance</u>: by averaging the prediction of all the models thus reducing the variance in the final prediction.

  Weakness:

  * In practice, the ensemble of trees tend to be highly <u>correlated</u>, because of their greedy nature, especially in the shallower nodes of the individual decision trees.

* **Why are decision trees greedy?**

  Decision trees are <u>NP-complete</u>, there is no way to find the global minima i.e. the best tree unless we use brute force and try all possible combinations. In practice this is infeasible. Consequently, practical decision-tree learning algorithms are based on <u>heuristic algorithms such as the greedy algorithm where locally optimal decisions are made at each node</u>. Hueristic algorithms are designed to solve problems in a faster more efficient method by sacrificing optimality, accuracy or precision in favor of speed. Hueristic algorithms are often used to solve NP-complete problems. Such algorithms <u>cannot guarantee to return the globally optimal decision tree</u>. This can be mitigated by training multiple trees in an ensemble learner, where the features and samples are randomly sampled with replacement (Bagging).

* **What is random forest? How does it differ from Bagging?** 

  Random Forest is a modified form of bagging that creates ensembles of independent decision trees. To <u>de-correlated</u> the trees, we:

  1. train each tree on a separate bootstrap <u>random sample</u> of the full training set (<u>same as in bagging</u>)

     **Why?** <u>Stochasticity</u> is added by constructing each predictor on a bootstrap sample of the original training set. This approach generally reduces variance.

  2. for each tree, at each split, we <u>randomly select a set of $J'$ predictors from the full set of predictors</u>. (<u>not done in bagging</u>)

     **Why?** The <u>multicollinearity</u> problem is alleviated since a random subset of features is chosen for each tree in a random forest. Bagging suffers from feature correlation but random forest address this issue.

  3. From amongst the $J'$ predictors, we select the optimal predictor and the optimal corresponding threshold for the split.

* **Can random forest overfit?**

  No. <u>Random Forests do not overfit as a function of forest size</u>. The testing performance of Random Forests does not decrease (due to overfitting) as the <u>number of trees increases</u>. Hence after certain number of trees the performance tend to stay in a certain value. <u>Increasing the number of individual randomized models in an ensemble will never increase the generalization error.</u> 

  However, if the number of trees in the ensemble is too large, then <u>the trees in the ensemble may become correlated, and therefore increase the variance</u>.

  We usually don't need to optimize the tree depth (should probably be as large as possible), but may want to customize stopping criterion based on the number of observations in each node.

* **What the "feature importance" given by random forest?**

  Feature importance is calculated as the <u>decrease in node impurity weighted by the probability of reaching that node</u>. The node probability can be calculated by the number of samples that reach the node, divided by the total number of samples. The higher the value the more important the feature.

* **When can random forest fail?**

  When we have a lot of predictors that are completely independent of the response and one overwhelmingly influential predictor.

* **Are bagging or random forest models independent of each other, can they be trained in a parallel fashion?**

  Yes. They train a bunch of individual / independent models in a parallel way. Each model is trained by a random subset of data.

* **What is Boosting?**

  Boosting is a sequential process, where each subsequent model attempts to correct the errors of the previous model. The succeeding models are dependent on the previous model. In this technique, learners are learned sequentially with early learners fitting simple models to the data and then analyzing data for errors. In other words, we fit consecutive trees (random sample) and at every step, the goal is to solve for net error from the prior tree. When an input is misclassified by a hypothesis, its weight is increased so that next hypothesis is more likely to classify it correctly. By combining the whole set at the end converts weak learners into better performing model.

  It trains a large number of "weak" learners <u>in sequence</u>. A weak learner is a constrained model (limit the max depth of each decision tree). Each one in the sequence focuses on <u>learning from the mistakes of the one before it</u>. By <u>more heavily weighting in the mistakes in the next tree</u> (the correct predictions are identified and less heavily weighted in the next tree), our next tree will learn from the mistakes. Boosting additively combines all the weak learners into a single strong learner, and gets a boosted tree. The ensemble is a <u>linear combination</u> of the simple trees, and is more expressive.

  Steps:

  1. A subset is created from the original dataset. Initially, all data points are given equal weights. A base model is created on this subset. This model is used to make predictions on the whole dataset.
  2. Errors are calculated using the actual values and predicted values. The observations which are incorrectly predicted, are given higher weights. Another model is created and predictions are made on the dataset. This model tries to correct the errors from the previous model. 
  3. Similarly, multiple models are created, each correcting the errors of the previous model. The final model (strong learner) is the weighted mean of all the models (weak learners).

* **Compare bagging and boosting?**

  https://www.kaggle.com/prashant111/bagging-vs-boosting

  Main Differences:

  * Bootstrapping
    - Bagging and Boosting get N learners by generating additional data in the training stage.
    - N new training data sets are produced by random sampling with replacement from the original set.
    - By sampling with replacement some observations may be repeated in each new training data set.
    - In the case of Bagging, any element has the same probability to appear in a new data set.
    - However, for Boosting the observations are weighted and therefore some of them will take part in the new sets more often.
    - These multiple sets are used to train the same learner algorithm and therefore different classifiers are produced.
  * Boosting builds the new learner in a sequential way (while bagging in parallel)
    * In Boosting algorithms each classifier is trained on data, taking into account the previous classifiers’ success.
    * After each training step, the weights are redistributed. Misclassified data increases its weights to emphasise the most difficult cases.
    * In this way, subsequent learners will focus on them during their training.
  * Classification
    * To predict the class of new data we only need to apply the N learners to the new observations. In Bagging the result is obtained by averaging the responses of the N learners (or majority vote). However, Boosting assigns a second set of weights, this time for the N classifiers, in order to take a weighted average of their estimates.
    * In the Boosting training stage, the algorithm allocates weights to each resulting model. A learner with good a classification result on the training data will be assigned a higher weight than a poor one. So when evaluating a new learner, Boosting needs to keep track of learners’ errors, too.
    * Some of the Boosting techniques include an extra-condition to keep or discard a single learner. For example, in AdaBoost, the most renowned, an error less than 50% is required to maintain the model; otherwise, the iteration is repeated until achieving a learner better than a random guess.

* **Similarities and Differences between bagging and boosting?**

  Similarity:

  * Both are ensemble methods to get N learners from 1 learner.
  * Both generate several training data sets by random sampling. 
  * Both make the final decision by averaging the N learners (or taking the majority of them i.e Majority Voting).
  * Both are good at reducing variance and provide higher stability.

  Differences:

  * **Bagging** is the simplest way of combining predictions that belong to the same type, while **Boosting** is a way of combining predictions that belong to the different types.
  * **Bagging** aims to decrease variance, not bias, while **Boosting** aims to decrease bias, not variance.
  * In **Bagging** each model receives equal weight, whereas in **Boosting** models are weighted according to their performance.
  * In **Bagging** each model is built independently, whereas in **Boosting** new models are influenced by performance of previously built models.
  * In **Bagging** different training data subsets are randomly drawn with replacement from the entire training dataset. In **Boosting** every new subsets contains the elements that were misclassified by previous models.
  * **Bagging** tries to solve over-fitting problem, while **Boosting** tries to reduce bias.
  * If the classifier is unstable (high variance), then we should apply **Bagging**. If the classifier is stable and simple (high bias) then we should apply **Boosting**.
  * **Bagging** is extended to Random forest model, while **Boosting** is extended to **Gradient boosting**.

* **Compare boosting trees and random forest?**

  <u>Boosting trees</u>:

  * All weak learners / trees are built in sequence by using iteration algorithm. Each weak learner / tree has very few leaves.
  * Boosting random samples the data <u>without replacement</u>.
  * Reassign weights to samples based on the results of previous iterations of classifications, mistakes are given more weights, correct predictions are given less weights, so it can learn from mistakes. 
  * Boosting performs bias reduction of simple trees which are easily underfitting (making simple trees more expressive).

  <u>Random forest</u>:

  * Is a special case of Bagging. A unique characteristic is that each tree randomly selects a subset of features. <u>Randomly</u> finds root node and split feature. The feature importance is calculated as the decrease in node impurity weighted by the probability of reaching that node.
  * Random forest uses bootstrap technique to random sample the data <u>with replacement</u>.
  * All trees are built in parallel. The final model take the majority for classification or average for regression.
  * Bagging performs variance reduction on complex trees which are easily overfitting.

* **Bagging or Boosting?**

  It depends on the problem and the data.

  Both bagging and boosting decrease the variance of your single estimate as they combine several estimates from different models. So the result may be a model with higher stability and robustness.
  
  If the problem is that the single model gets a very low performance, bagging will rarely be a good choice to reduce bias. However, boosting could generate a combined model with lower errors.
  
  If the problem is that the single model is overfitting, then bagging is the better option. Boosting does not help to avoid overfitting.
  
  Since boosting is faced with overfitting problems, bagging is effective more often than boosting.
  
* **What is AdaBoost?**

  AdaBoost is the first designed boosting algorithm with a particular loss function. 

  AdaBoost is adaptive boosting. It is adaptive in the sense that <u>subsequent weak learners are tweaked in favor of those instances misclassified by previous classifiers</u>. The model is adapted to put more weight on misclassified samples and less weight on correctly classified samples. The final prediction is a <u>weighted average of all the weak learners, where more weight is placed on stronger learners</u>.

  The technique of Boosting uses various loss functions. In case of AdaBoost, it minimizes the exponential loss function that can make the algorithm sensitive to the outliers.

* **How does AdaBoost get different splits in each weak learner?**

  The sample weights are updated after each iteration. It's the different sample weights that will cause splits to vary across our base estimators.

* **What is Gradient Boosting?**

  Gradient boosting looks at the difference between its current approximation, and the known correct target vector, which is called the <u>residual</u>. It fits the next model to the residuals of the current model.

* **Are Boosted models independent of one another? Do they need to wait for the previous model's residuals?**

  No they are dependent. The output for the new model is added to the output of the existing sequence of models in an effort to correct or improve the final output of the model. 

  Yes Gradient Boosting need residuals to minimize loss function by gradient descent. We can interpret the residuals as negative gradients of the loss function (if the loss function is defined as 1/2 squared residuals).

* **Compare AdaBoost and Gradient Boosting?**

  * <u>Loss function</u>: 
    * AdaBoost is regarded as a special case of Gradient Boosting in terms of loss function. AdaBoost minimizes a particular loss function - exponential loss function that can make the algorithm sensitive to the outliers. 
    * Gradient Boosting is a generic algorithm which is more flexible. Any differentiable loss function can be used.
  * <u>Strategy of improving models</u>: 
    * In AdaBoost, "shortcomings" are identified by <u>high-weight data points</u>. At each iteration, AdaBoost <u>changes the sample distribution</u> by modifying the weights attached to each of the instances. It increases the weights of the wrongly predicted instances and decreases the ones of the correctly predicted instances. The weak learner thus focuses more on the difficult instances. 
    * In Gradient Boosting, "shortcomings" are identified by <u>gradients</u>. The weak learner <u>trains on the remaining errors</u> (so-called pseudo-residuals) of the strong learner. It is another way to give more importance to the difficult instances. At each iteration, the pseudo-residuals are computed and a weak learner is fitted to these pseudo-residuals.
    * Both high-weight data points and gradients tell us how to improve the model.
  * <u>Strategy of adding models</u>: 
    * For AdaBoost, the final prediction is based on a <u>majority vote</u> of the weak learners’ predictions. The weak learners are added sequentially to the strong one weighted by their individual <u>accuracy / performance</u>  (so-called alpha weight). The higher it performs, the more it contributes to the strong learner.
    * All the learners have <u>equal weights</u> in the case of Gradient Boosting. The weight is usually set as the <u>learning rate</u> which is small in magnitude. The contribution of the weak learner is made by <u>minimizing the overall error of the strong learner</u>, by gradient descent optimization process.
  * <u>Tasks</u>: 
    * AdaBoost was mainly designed for <u>binary classification</u> problems and can be utilised to boost the performance of decision trees.
    * Gradient Boosting is used to solve the <u>differentiable loss function</u> problem, therefore it can be used for <u>both classification and regression problems</u>. 

* **What is XGBoost and Why XGBoost?**

  XGBoost is a decision-tree-based ensemble machine learning algorithm that uses a gradient boosting framework.

  <u>Accuracy:</u>

  - XGBoost uses a <u>more regularized model formalization to control overfitting</u> (=better performance) by both L1 and L2 regularization.
  - Tree Pruning methods: more shallow tree will also prevent overfitting
  - Improved convergence techniques (like early stopping when no improvement is made for X number of iterations)
  - Built-in Cross-Validaiton

  <u>Computing Speed:</u>

  - Special Vector and matrix type data structures for faster results.
  - <u>Parallelized tree building</u>: using all of your CPU cores during training.
  - <u>Distributed Computing</u>: for training very large models using a cluster of machines.
  - Cache Optimization of data structures and algorithm: to make best use of hardware.

* **Does XGBoost build boosted trees in parallel?**

  No. XGBoost doesn't run multiple trees in parallel, it needs predictions after each tree to update gradients. Rather it does the <u>parallelization WITHIN a single tree</u> by using openMP to create branches independently.

* **What is bias-variance tradeoff and how to solve it?**

  - The <u>bias</u> of a model quantifies how precise a model is across training sets.
  - The <u>variance</u> quantifies how sensitive the model is to small changes in the training set.
  - A <u>robust</u> model is not overly sensitive to small changes.
  - <u>The dilemma involves minimizing both bias and variance</u>; we want a precise and robust model. Simpler models tend to be less accurate but more robust. Complex models tend to be more accurate but less robust.

  <u>How to reduce bias:</u>

  - Use more complex models, more features, less regularization, ...
  - <u>Boosting</u>: attempts to improve the predictive flexibility of simple models. Boosting uses simple base models and tries to “boost” their aggregate complexity.

  <u>How to reduce variance:</u>

  - <u>Early Stopping:</u> Its rules provide us with guidance as to how many iterations can be run before the learner begins to over-fit.
  - <u>Pruning</u>: Pruning is extensively used while building related models. It simply removes the nodes which add little predictive power for the problem in hand.
  - <u>Regularization</u>: It introduces a cost term for bringing in more features with the objective function. Hence it tries to push the coefficients for many variables to zero and hence reduce cost term.
  - <u>Train with more data:</u> It won’t work every time, but training with more data can help algorithms detect the signal better.
  - <u>Bagging</u>: attempts to reduce the chance of overfitting complex models: Bagging uses complex base models and tries to “smooth out” their predictions.
  
* **What is the impact of having too many trees in boosting and in bagging? In which instance is it worse to have too many trees?**

  Boosting can overfit the training data if it uses too many trees, because the construction of each tree depends strongly on the trees that have already been grown. 

  While bagging overfits very slowly when the number of trees increases (it just increases the variance slowly).

* **Which of the bagging and boosting can be extended to regression task? How?**

  Both bagging and boosting can be extended to regression tasks. To apply bagged regression trees, we construct B regression trees using B bootstrapped training sets and then average the resulting prediction. To apply boosted regression trees, we first fit a decision tree to the residuals from the model as the response. We then add this new decision tree into the fitted function in order to update the residuals. By fitting small trees to the residuals, we slowly improve the target in areas where it does not perform well.

* **Difference between ensemble methods and mixture of experts?**

  Ensemble methods and mixture of experts combine models in order to obtain a more accurate and / or more robust model.

  * <u>Ensemble methods</u> combines models cooperatively

    * Homogeneous models

      * Voting and averaging
      * Bagging and boosting

    * Heterogeneous models

      * Blending

        Independent, parallel, heterogeneous weak learners, by training a meta-model to output a prediction - less bias

      * Stacking

        Similar to blending, except each model is now used to make out of distribution predictions

  * <u>Mixture of experts</u> combines models by specialization

    Uses multiple simple learners, each of which specializes on a different part of the data, plus a manager model that will decide which specialist to use for each input data.

    Good if the dataset contains several different regimes which have different relationships between input and output. Covers different input regions with different learners.

    Learning involves learning the parameters of each expert and the parameters of the gating network.

    * Heterogeneous models
      * Experts and (softmax) gating network 
  
  
