# Machine Learning Concepts Q&A

## 1. Regression

* **We have had a train-test split. Why we need a validation set?** 

  The *training set* is the data we have seen. We train different types of models with various different hyperparameters and regularization parameters on this data.

  The *validation set* is used to <u>compare different models</u>. We use this set to <u>tune our hyper-parameters</u> i.e. find the optimal set of hyper-parameters and pick the best model.

  The *test set* is used to report the scores using our best model. <u>We don't tune parameters using test set</u>, because we want to know the generalizability of our model, i.e. how our model performs on a new data set.

* **For which models do the unit / scale of the predictors matter?**

  <u>k-NN</u>. Scaling affects distance metric - <u>Euclidean distance measure</u>, which determines what '"neighbor" means. All such distance based algorithms are affected by the scale / magnitude of the variables.

  <u>Clustering</u>. It is important to standardize variables before running Cluster Analysis. It is because <u>cluster analysis techniques depend on the concept of measuring the distance between the different observations we're trying to cluster</u>. If a variable is measured at a higher scale than the other variables, then whatever measure we use will be overly influenced by that variable.

  <u>PCA</u>. Prior to PCA, it is critical to standardize variables. It is because <u>PCA gives more weight to those variables that have higher variances</u> than to those variables that have very low variances. In effect the results of the analysis will depend on what units of measurement are used to measure each variable. Standardizing raw values makes equal variance so high weight is not assigned to variables having higher variances.

  <u>SVM</u>. <u>All SVM kernel methods are based on distance</u> so it is required to scale variables prior to running final Support Vector Machine (SVM) model.

  <u>Lasso and Ridge</u>. <u>They put constraints on the size of the coefficients associated to each variable</u>. However, this value will depend on the magnitude of each variable.

  Scaling does not affect statistical inference in simple regression models, as the estimates are adjusted appropriately and the p-values will be the same. But it does <u>removes collinearity</u>. Interaction term would introduced collinearity. Scaling the one of the two variables can remove collinearity.

* **Compare normalization and standardization?**

  The point of normalization and standardization is to make the variables take on a more manageable scale. Ideally, we should standardize or normalize all the variables at the same time. But they are not always needed.

  Normalization makes data go from 0 to 1. It is widely used in image processing and computer vision, where pixel intensities are non-negative and are typically scaled from a 0-255 scale to a 0-1 range for a lot of different algorithms.

  Normalization is very useful in neural networks as it leads to the algorithms converging faster.

  Normalization is useful when the data <u>does not have a discernible distribution</u> and we are not making assumptions about the data's distribution.

  <u>Standardization maintains outliers</u> whereas normalization makes outliers less obvious. Standardization is preferred when the outliers are useful.

  Standardization is useful when we assume the data come from a <u>Gaussian distribution</u> (approximately).

  standardization makes it realistic to interpret the  intercept term as the expected value of $Y_i$ when the predictor values are set to their means.

* **What is Bootstrapping?**

  Bootstrapping is a procedure for resampling a dataset <u>with replacement</u> to produce an <u>empirical distribution</u> of the value of interest.

  The key points of Bootstrapping: 

  1. We need to preserve key statistics about the original distribution to come up with a good empirical distribution. 
  2. We must sample with replacement.

* **What is confidence interval and how to interpret it?**

  A confidence interval is a range of values that is likely to include a parameter of interest with some degree of certainty or "confidence."

  If we were to compute 95% confidence intervals for each of K repeated samples, we would expect <u>0.95*K of those confidence intervals to contain the true parameter of interest</u>.

* **What is multicollinearity?**
  Multicollinearity occurs when independent variables in a regression model are correlated. This correlation is a problem because independent variables should be independent. If the degree of correlation between variables is high enough, it can cause problems when you fit the model and interpret the results.

  Multicollinearity is a problem because <u>it undermines the statistical significance of an independent variable</u>. If two independent variables are correlated, it is hard to distinguish the effect of these two variables on the dependent variable. Other things being equal, the larger the standard error of a regression coefficient, the less likely it is that this coefficient will be statistically significant. 

* **What does regularization help with?**

  We have some pretty <u>large and extreme coefficient values with high variance</u> in our models. We can clearly see some overfitting to the training set. In order to reduce the coefficients of our parameters, we can introduce a penalty term that penalizes some of these extreme coefficient values.

  1. <u>Avoid overfitting</u>. Reduce features that have weak predictive power. (Idea: reduce variance by increase bias)
  2. Discourage the use of a model that is too complex.

* **Compare Ridge and Lasso regression?**

  Ridge regression reduces the complexity of the model by <u>shrinking the coefficients</u>, but it doesn’t nullify them. It controls the amount of regularization using a parameter $\lambda$. <u>The penalty term is proportional to the L2-norm of the coefficients</u>.

  Lasso regression controls the amount of regularization using a parameter $\lambda$. It controls the amount of regularization using a parameter $\lambda$. <u>The penalty term is proportional to the L1-norm of the coefficients</u>.

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

* **What is the benefits of ROC curve and AUC score?**

  The ROC curve allows us to find the classification threshold that gives the best FPR and TPR trade-off. 

  By summarizing the information in the ROC curve, we can compare our classifier against a perfect classifier and a random classifier. 

  We can compare models by comparing AUC score, which is the area under the ROC curve.

* **What is decision tree? Why decision tree?** 

  Decision tree is a graphical representation of possible solutions to a decision based on certain conditions.

  It's like simple flow charts that can be formulated as mathematical models for classification and these models have the properties we desire;

  - Interpretable by humans
  - Have sufficiently complex decision boundaries
  - The decision boundaries are locally linear, each component of the decision boundary is simple to describe mathematically.

* **Difference of decision tree on classification and regression?**

  For classification, we return the <u>majority</u> class in the points of each leaf node.

  For regression we return the <u>average</u> of the outputs for the points in each leaf node.

* **How does a Decision Tree decide on its splits (what is the criteria for a split point)?**

  It finds the feature that best splits the target class into the purest possible children nodes. Eventually, decision tree aims to achieve the minimal average classification error.

  The measure of purity is called the <u>information</u>. It represents the <u>expected amount of information that would be needed to specify whether a new instance should be classified 0 or 1, given the example that reached the node</u>.

  <u>Entropy</u> on the other hand is a measure of impurity. The formula is $-p(a) * \log (p(a)) -p(b) * \log (p(b)) $. By comparing the entropy before and after the split, we obtain the information gain, it is how much the information gained by doing the split using that particular feature. So the information gain can be calculate by entropy before the split minus the entropy after the split.

  <u>Gini index</u> is another commonly used measure of purity. The formula of gini index is 1 minus the <u>squared probability</u> of each class. The optimal feature to split is chosen by minimizing the <u>weighted sum</u> of Gini index in each child node.

* **What is "stopping condition" in a decision tree?**

  To avoid overfitting, we could

  - Stop the algorithm at a particular depth. (=<u>not too deep</u>)
  - Don't split a region if all instances in the region belong to the same class. (=<u>stop when subtree is pure</u>)
  - Don't split a region if the number of instances in the sub-region will fall below pre-defined threshold (min_samples_leaf). (=<u>not too specific/small subtree</u>)
  - Don't use too many splits in the tree (=<u>not too many splits / not too complex global tree</u>)
  - Be content with <100% accuracy training set...
  
* **Why is using FPR and TPR not as good as precision and recall for evaluating models with unbalanced data?**

## 4. Ensemble

* **What is bagging? Why (not) bagging?** 

  Bagging is an <u>ensemble</u> meta-algorithm combining predictions from multiple-decision trees through a <u>majority voting</u> mechanism.

  Bagging is a <u>greedy</u> algorithm. We always choose the feature with the most impact: i.e. the most informative gain.

  The key idea is: One way to adjust for the <u>high variance</u> of the output of an experiment is to perform the experiment multiple times and then average the results.

  1. <u>Bootstrap</u>: we generate multiple samples of training data, via bootstrapping. We train a full decision tree on each sample of data.
  2. <u>AGgregatiING</u>: for a given input, we output the averaged outputs of all the models for that input.

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

* **What is Boosting?**

  It trains a large number of “weak” learners <u>in sequence</u>. A weak learner is a constrained model (limit the max depth of each decision tree). Each one in the sequence focuses on <u>learning from the mistakes of the one before it</u>. By <u>more heavily weighting in the mistakes in the next tree</u> (the correct predictions are identified and less heavily weighted in the next tree), our next tree will learn from the mistakes. Boosting additively combines all the weak learners into a single strong learner, and gets a boosted tree. The ensemble is a <u>linear combination</u> of the simple trees, and is more expressive.

* **Are bagging or random forest models independent of each other, can they be trained in a parallel fashion?**

  Yes. They train a bunch of individual / independent models in a parallel way. Each model is trained by a random subset of data.

* **Compare bagging and boosting?**

  https://www.kaggle.com/prashant111/bagging-vs-boosting

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

* **What is AdaBoost?**

  AdaBoost is adaptive boosting. It is adaptive in the sense that <u>subsequent weak learners are tweaked in favor of those instances misclassified by previous classifiers</u>. The model is adapted to put more weight on misclassified samples and less weight on correctly classified samples. The final prediction is a <u>weighted average of all the weak learners, where more weight is placed on stronger learners</u>. Eventually, Adaboost minimizes a differentiable error function (<u>exponential loss</u>: differentiable with respect to $\widehat{y}$ and it is an <u>upper bound of error</u>).

* **How does AdaBoost get different splits in each weak learner?**

  The sample weights are updated after each iteration. It's the different sample weights that will cause splits to vary across our base estimators.

* **What is Gradient Boosting?**

  Gradient boosting looks at the difference between its current approximation, and the known correct target vector, which is called the <u>residual</u>. It fits the next model to the residuals of the current model.

* **Are Boosted models independent of one another? Do they need to wait for the previous model's residuals?**

  No they are dependent. The output for the new model is added to the output of the existing sequence of models in an effort to correct or improve the final output of the model. 

  Yes Gradient Boosting need residuals to minimize loss function by gradient descent. We can interpret the residuals as negative gradients of the loss function (if the loss function is defined as 1/2 squared residuals).

* **Compare AdaBoost and Gradient Boosting?**

  * <u>Loss function</u>: 
    * AdaBoost is regarded as a special case of Gradient Boosting in terms of loss function. AdaBoost minimizes a particular loss function - exponential error function. 
    * Gradient Boosting is a generic algorithm which is more flexible. For any loss function, we can derive a gradient boosting algorithm.
  * <u>Strategy of improving models</u>: 
    * In AdaBoost, "shortcomings" are identified by <u>high-weight data points</u>. At each iteration, AdaBoost <u>changes the sample distribution</u> by modifying the weights attached to each of the instances. It increases the weights of the wrongly predicted instances and decreases the ones of the correctly predicted instances. The weak learner thus focuses more on the difficult instances. 
    * In Gradient Boosting, "shortcomings" are identified by <u>gradients</u>. The weak learner <u>trains on the remaining errors</u> (so-called pseudo-residuals) of the strong learner. It is another way to give more importance to the difficult instances. At each iteration, the pseudo-residuals are computed and a weak learner is fitted to these pseudo-residuals.
    * Both high-weight data points and gradients tell us how to improve the model.
  * <u>Strategy of adding models</u>: 
    * All the learners have <u>equal weights</u> in the case of Gradient Boosting. The weight is usually set as the <u>learning rate</u> which is small in magnitude. The contribution of the weak learner is made by <u>minimizing the overall error of the strong learner</u>, by gradient descent optimization process.
    * For AdaBoost, the final prediction is based on a <u>majority vote</u> of the weak learners’ predictions. The weak learners are added sequentially to the strong one weighted by their individual <u>accuracy / performance</u>  (so-called alpha weight). The higher it performs, the more it contributes to the strong learner.
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
