# ML Interview Questions

1. **What are the main assumptions of a linear regression? what are the most common estimation techniques for linear regression?**

   A linear regression models the relationship between the dependent variable Y and the independent variable X.

   The four assumptions are: (LINE)

   1. <u>Linearity</u>: The relationship between X and Y is linear.
   2. <u>Independence</u>: All responses Y are independent of each other.
   3. <u>Normality</u>: The residuals are normally distributed.
   4. <u>Equality</u>: The responses Y have equal variance about the mean for all independent variable X.

   Most common types of linear regression:

   * Ordinary least squares

   * Generalized least squares

   * Penalized least squares

     * <u>L1 (Lasso) and L2 (Ridge)</u>

       Lasso and Ridge regressions are similar to linear regression in that they also minimize the residual sum of squared (RSS), but they add a penalty term to avoid overfitting.

2. **Describe the formula for logistic regression and how the algorithm is used for binary classification.**

   The logit transformation of the odds of the positive class is linearly related to the independent variables. 

   The formula is $p = {1\over 1 + e^{-X\beta}}$, which can be used as the probability of the data point being in the positive class.

3. **How does a Decision Tree decide on its splits (what is the criteria for a split point)?**

   It finds the feature that best splits the target class into the purest possible children nodes. Eventually, decision tree aims to achieve the minimal average classification error.

   The measure of purity is called the <u>information</u>. It represents the <u>expected amount of information that would be needed to specify whether a new instance should be classified 0 or 1, given the example that reached the node</u>.

   <u>Entropy</u> on the other hand is a measure of impurity. The formula is $-p(a) * \log (p(a)) -p(b) * \log (p(b)) $. By comparing the entropy before and after the split, we obtain the information gain, it is how much the information gained by doing the split using that particular feature. So the information gain can be calculate by entropy before the split minus the entropy after the split.

   <u>Gini index</u> is another commonly used measure of purity. The formula of gini index is 1 minus the <u>squared probability</u> of each class. The optimal feature to split is chosen by minimizing the <u>weighted sum</u> of Gini index in each child node.

4. **What advantages does a decision tree model have over other machine learning models?**

   It's very easy to interpret and understand.

   It works nicely on both continuous and categorical features.

   No normalization or scaling is necessary, because each individual feature is considered separately from the other ones.

   The prediction algorithm runs very fast and can run on very large dataset.

5. **What is the difference between a random forest versus boosting tree algorithms?**

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

6. **Given a data set of features X and labels y , what assumptions are made when using Naive Bayes methods?**

   Naive Bayes algorithm assumes that the features of $X$ are <u>conditionally</u> independent of each other for the given $Y$.

7. **Describe how the support vector machine (SVM) algorithm works.**

   SVM attempts to find a hyperplane that separates classes by maximizing the margin. The points on the margin are called support vectors. So this is SVM for <u>linear classification</u>.

   SVM can also perform <u>nonlinear classification</u>. SVM can employ the kernel trick which can map linear non-separable inputs into a higher dimension where they become more easily separable.

8. **What is overfitting and what causes it?What ways can you attempt to avoid overfitting?**

   Overfitting is when the ML model does not generalize well to new data. When a model overfits to the training data, usually it means it's too complex.

   Underfitting is when a model perform badly on both training and test set. It has high bias and low variance. Overfitting on the other hand perform very well on the training set but perform badly on test set. It has high variance and low bias.

   Ways to fix overfitting:

   * <u>Regularization.</u>
   * <u>Dimensionality reduction.</u>
   * <u>K-fold cross validation.</u>
   * <u>Early stopping</u>
   * <u>Ensembling</u>

9. **Describe the differences between accuracy, sensitivity, specificity, precision, and recall.**

   <u>Accuracy</u> = ${TP + TN \over TP + TN + FP + FN }$ = proportion of accurate predictions among all predictions

   <u>Sensitivity</u> = <u>true positive rate</u> = <u>recall</u> = ${TP \over TP + FN}$

   <u>Specificity</u> = <u>false negative rate</u> =  1- false positive rate = ${TN \over TN + FP}$

   <u>Precision</u> = ${TP\over TP + FP}$

   <u>F1-score</u> = $2 * {precision * recall \over precision + recall }$

10. **Define precision and recall.**

    Recall is also known as the true positive rate: the amount of positives your model claims compared to the actual number of positives there are throughout the data. Precision is also known as the positive predictive value, and it is a measure of the amount of accurate positives your model claims compared to the number of positives it actually claims. 

11. **What metrics can be used to evaluate a regression task?**

		* <u>Mean absolute error</u> (MAE)
		* <u>Mean squared error</u> (MSE)
		* <u>Root mean squared error</u> (RMSE)

11. **What’s the trade-off between bias and variance?**

    Bias is error due to overly simple assumptions in the learning algorithm. It can lead to underfitting the data.

    Variance is error due to too complex in learning algorithm. It can lead to overfitting the data.

    The trade-off is that by making the model more expressive and complex, the model will have higher variance and lower bias. By simplifying the model, it will have higher bias and lower variance.

12. **What is the difference between supervised and unsupervised machine learning?**

    Supervised learning requires training labeled data. Unsupervised learning, in contrast, does not require labeling data explicitly.

13. **How is KNN different from k-means clustering?**

    K-Nearest Neighbors is a supervised classification algorithm, while k-means clustering is an unsupervised clustering algorithm.

    *k*-means clustering aims to partition the *n* observations into *k* (≤ *n*) sets, so as to minimize the within-cluster sum of squares (WCSS).

    kNN assigns a point to the class of its closest neighbor in the feature space, or assigns a point to the majority voted class of its closest k neighbors in the feature space.

14. **Explain how a ROC curve works.**

    ROC curve is plotted by the true positive rate versus the false positive rate at various threshold.

15. **What is AI, ML, DL?**

    Artificial intelligence describes a machine that mimics human behavior in some way. AI can make the user experience similar to interacting with a human. The human part is the output. The input is huge amounts of data, which allows the AI to learn and adapt. AI includes ML, DL, and NN.

    Machine learning is a subset of AI and is a set of techniques that give computers the ability to learn without being explicitly programmed to do so.

    Deep learning is a further subset of machine learning that enables computers to learn from complex patterns and solve more complex problems. All of the deep learning models are based on neural networks.

16. **What is Bayes’ Theorem? How is it useful in a machine learning context?**

    Bayes’ Theorem gives you the posterior probability of an event given what is known as prior knowledge.

17. **Why is “Naive” Bayes naive?**

    Naive Bayes is considered “Naive” because it makes an assumption that is virtually impossible to see in real-life data: the conditional probability is calculated as the pure product of the individual probabilities of components. In other words, the assumption is that all the features are conditionally independent given Y. Hence this algorithm is called naive because it makes very naive or crude assumption about the features.

18. **Explain the difference between L1 and L2 regularization.**

    L1 and L2 regularization allows to minimize the MSE and a penalty term at the same time. The penalty term in L1 regularization is proportional to the L1 norm of coefficients. L1 Penalization works well when dealing with high dimensional data because its penalty parameter λ allows us to ignore or remove irrelevant features.

    The penalty term in L2 regularization is proportional to the L2 norm of coefficients. L2 regularization allows us avoid overfitting by shrinking large coefficients towards zero.

19. **What’s your favorite algorithm, and can you explain it to me in less than a minute?**

    XGBoost is my favorite. It stands for eXtreme Gradient Boosting. XGBoost is a decision-tree-based ensemble machine learning algorithm that uses a gradient boosting framework. It trains weak learner / tree sequentially on subsampled (without replacement) data, using gradient descent algorithm to minimize the overall MSE. It has several nice properties:

    * Add regularization to avoid overfitting.
    * Built-in cross validation.
    * Parallel and distributed computing.

20. **What’s the difference between Type I and Type II error?**

    Type I error is the probability of rejecting the null hypothesis given the null hypothesis is true. It is equal to false positive. Type II error is the probability of not rejecting the null hypothesis given the null hypothesis is not true. It is equal to false negative.

11. **What’s a Fourier transform?**

    https://betterexplained.com/articles/an-interactive-guide-to-the-fourier-transform/

12. **What’s the difference between probability and likelihood?**

    The likelihood is a function of parameter given observed data. The probability is a function of random variable given parameter.

13. **What is deep learning, and how does it contrast with other machine learning algorithms?**

    Deep learning is a subset of machine learning that is concerned with neural networks. NN uses multiple layers to progressively extract higher-level features from the raw input. 

    * Machine learning is about computers being able to think and act with less human intervention; deep learning is about computers learning to think using structures modeled on the human brain.
    * Deep learning can analyze images, videos, and unstructured data in ways machine learning can’t easily do.
    * Machine learning requires less computing power; deep learning typically needs less ongoing human intervention.

14. **What’s the difference between a generative and discriminative model?**

    A generative model learns the <u>joint probability distribution</u> $p(x,y)$ and a discriminative model learns the <u>conditional probability distribution</u> $p(y|x)$.

    Generative models model the <u>distribution</u> of individual classes, while discriminative models learn the <u>boundary</u> between classes.

    Generative models are typically specified as probabilistic graphical models, which offer rich representations of the independence relations in the dataset. Discriminative models focus on richly modeling the boundary between classes. Discriminative models will generally outperform generative models on classification tasks. Given the same amount of capacity (say, bits in a computer program executing the model), a discriminative model may yield more complex representations of this boundary than a generative model.

    * SVMs and decision trees are discriminative because they learn explicit boundaries between classes. SVM is a maximal margin classifier, meaning that it learns a decision boundary that maximizes the distance between samples of the two classes, given a kernel. The distance between a sample and the learned decision boundary can be used to make the SVM a "soft" classifier. DTs learn the decision boundary by recursively partitioning the space in a manner that maximizes the information gain (or another criterion).

15. 

# DS Interview Questions

1. **What is the Central Limit Theorem and why is it important?**

   In CLT, it states that if we sample from a population using a sufficiently large sample size, the mean of the samples (also known as the *sample population*) will be normally distributed (assuming true random sampling). What’s especially important is that this will be true regardless of the distribution of the original population.

   Suppose we are interested in estimating the average height among all people. In practice, it is not practical to collect all data from the population. But we can still sample some people from the population. So the question becomes, what can we say about the average height of the entire population given a single sample. CLT addresses this question exactly.

2. **What is sampling? What sampling methods do you know?**

   Data sampling is a statistical analysis technique used to select, manipulate and analyze a representative subset of data points to identify patterns and trends in the larger data set.

   Probability based sampling includes: 

   * Simple random sampling
   * Stratified sampling
   * Cluster sampling
   * Multistage sampling
   * Systematic sampling

   Nonprobability based sampling includes

   * Convenience sampling
   * Consecutive sampling
   * Purposive or judgmental sampling
   * Quota sampling

3. **What is selection bias?**

   Active selection bias occurs when a subset of the data are systematically (i.e., non-randomly) excluded from analysis.