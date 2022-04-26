# Machine Learning Concept Q&A (II)

## 1. Unsupervised Learning & Cluster Analysis

* **What is unsupervised learning? Give some examples.**

  Unsupervised learning aims to discover subgroups among variables or observations, but not about how variables or observations relate to a response variable. It can be used as a descriptive tool.

  Examples include: to find subgroups of breast cancer patients according to similar gene expressions; to find subgroups among online shoppers according to their browsing and purchasing patterns; to find subgroups of movies according to ratings assigned by movie viewers.

* **What is the difference between PCA and clustering?**

  <u>PCA</u> attempts to find a low-dimensional representation of the observations that explains a large fraction of the variation, often used for dimensionality reduction, also be useful for visualization of grouping of data.

  <u>Clustering</u> aims to find homogeneous subgroups among the observations.

* **What is clustering? **

  Cluster analysis aims to find subgroups or clusters in a multivariate dataset. It partitions the data into distinct groups so that observations within each group are similar to each other.  It is usually domain-specific to concretely define what it means for observations to be similar or different.

  Clustering involves two steps 1) calculating inter-observation distances 2) applying clustering algorithm.

* **What are the two main types of approaches to clustering?**

    Two main types of approaches to clustering are

  1. <u>Partitioning clustering</u>

     Partitioning clustering specifies the number of clusters in advance, and then invoke an algorithm to partition the data. 

  2. <u>Hierarchical clustering</u>

     Hierarchical clustering iteratively merges or divides the data typically one observation at the time, and then decide on partitions afterward

* **What is the basic idea of partitioning clustering?**

  The basic idea is to specify the number of clusters into which the data will be partitioned, and then perform computation to group data so that

  * Observations within clusters are similar (low distances / dissimilarities), and
  * Observations in different clusters are dissimilar (high distances / dissimilarities).

* **Why distance measures are important to clustering? **

  Distance measures are important because 1) the pairwise distance calculation can influence the shape of the clusters, and 2) computing distances first followed by clustering can be efficient when observations to be clustered can be non-standard such as pictures and audio signals.

* **Compare some common distance measures?**

  * Two common distance measures: for observations $x_i$ and $x_k$ (of length $p$).

    * <u>Euclidean distance (L2)</u>:  $d(x_i, x_k) = \sqrt{\sum^p_{j=1} (X_{ij} - X_{kj})^2}$
      * sum of squares pairwise distances of ith and kth observation w.r.t jth feature, then take the square root at the end.
      * Sensitive to outliers; more efficient for normally distributed data.
    * <u>Manhattan distance (L1)</u>:  $d(x_i, x_k) = \sum^p_{i=1}|X_{ij} - X_{kj}|$
      * sum of absolute value of pairwise distance of ith and kth observation w.r.t jth feature.
      * Robust to outliers.

  * Other distance measures: correlation-based measures (measures how linearly related the set of variables are)

    * <u>Pearson distance</u>
      $$
      d(x_i, x_k) = 1 - {\sum^p_{j=1} (X_{ij} - \overline{x_i})(X_{kj} - \overline{x_k})\over \sqrt{\sum^p_{j=1}(X_{ij} - \overline{x_i})^2 \sum^p_{j=1}(X_{kj} - \overline{x_k})^2}}
      $$

      * The second part is the correlation between 2 observations.

      * Since the range of correlation is $ [-1, +1]$, the range of Pearson distance is $[0,2]$.
      * Sensitive to outliers.

    * <u>Spearman distance</u>
      $$
      d(x_i, x_k) = 1 - {\sum^p_{j=1} (W_{ij} - \overline{x_i})(W_{kj} - \overline{w_k})\over \sqrt{\sum^p_{j=1}(W_{ij} - \overline{w_i})^2 \sum^p_{j=1}(W_{kj} - \overline{w_k})^2}}
      $$

      * It replaces each value of a column by the ranks, for all columns. $W_{ij}$ are the ranks of $X_{ij}$ for feature $j$, and $\overline{w_i}$ is the average of the ranks for observation $i$. 
      * Robust to outliers. Use when outliers might be a concern in the data.

* **How to choose distance measures?**

  Beside the special properties of each distance measure that need to be considered, it actually does not matter whether one measure is larger than the other for a dataset. What is matter is we apply it consistently with in a dataset. The choice of measures should self-consistent within the analysis.

* **Why it is necessary to scaling before computing inter-distance measures between each features?** 

  If one feature has large values, when we compute distances, the distance between any two observation is dominantly determined by that feature, and we can hardly see the distances by other features. This is an undesired consequence of variables having different scales. Usually, we want to ensure each variable has equal contribution to the distance computation for clustering.

  The solution is <u>to standardize the variables prior to computing distances</u>. A common way is to transform each value by 1) subtracting the sample mean or median of the jth variable from $X_{ij}$ 2) divided by teh sample standard deviation or mean absolute deviation of the jth variable. (Usually use 'mean' when there is no skewness of outliers).
  $$
  {X_{ij} - \text{center}(x_j)\over \text{scale}(x_j)}
  $$

* **What is biplot of PCA analysis?**

  A biplot is a scatter plot of the first two principal components obtained from PCA analysis. It displays both the PC scores and the PC loadings (PC loadings are the directions of some different variables related to the first and second PC).

* **What is k-means clustering? Goal and results? **

  The resulting clusters have two properties:

  * The combination of all clusters is the whole set of data.
  * All clusters are mutually exclusive.

  The mean goal of K-means clustering is to determine K clusters such that the <u>total sum of within-cluster variation $\sum^K_{k=1} W(C_k)$ is minimized</u>. This value is sometimes called <u>inertia</u>.
  
  There are two ways to computing <u>within-cluster variation</u> for a particular cluster K:
  
    * <u>Average of pairwise distances within cluster K</u>.
      $$
      W(C_k) = {1\over |C_k|} \sum_{i,i' \in C_k} \sum^p_{j=1}(X_{ij} - X_{i'j})^2
      $$
  
      * $\sum^p_{j=1}(X_{ij} - X_{i'j})^2$: sum of distances between ith and i'th observation for each individual feature.
      * $\sum_{i,i' \in C_k} \sum^p_{j=1}(X_{ij} - X_{i'j})^2$: sum of pairwise distances within cluster K. Within each cluster, we calculate the Euclidean distance between every pair of observations, and then sum them up.
      * $C_k$: number of data points in cluster K.
  
    * <u>Twice the within-cluster sums-of-squares</u> (mathly derived from the previous formula)
      $$
      W(C_k) = 2 \sum_{i,i' \in C_k} \sum^p_{j=1}(X_{ij} - \overline{x_{jk}})^2
      $$
  
      * $\overline{x_{jk}}$: is the sample mean of the $X_{ij}$ for $i \in C_k$, or say, the variable-wise average of the observations in cluster k.

* **What is the k-means algorithm?**
  1. <u>Randomly assign</u> each observation to one of K clusters at random.
  2. Repeat the following two steps until clusters do not change:
     * For each cluster k, compute the <u>cluster centroid</u> $\overline{x_k}$ (<u>The variable-wise average of the observations in cluster k</u>).
     * Given the K centroids, <u>reassign all observations to clusters based on their closeness to the centroids</u>.

* **The bad and the good of k-means clustering? And solutions to the bad?**
    
    * It requires analyst to pre-specify K in advance. (Though there are ways to optimize K)
      * Try various values of K and compare results
      * Try different initial cluster assignments in parallel, and then choose the solution with the best <u>within-cluster sum of squared deviations</u>.
    * The algorithm is <u>locally optimal</u>, not globally optimal. Therefore, we can get different clusterings depending on the starting cluster assignment.
    
* **How to use and interpret Silhouette plot?**

  Silhouette plot is a diagnostic method for any clustering including k-means. Once a clustering has been determined, we can calculate <u>silhouette</u> for observation $i$ as:
  $$
  s_i = {b_i - a_i \over \max(a_i, b_i)}
  $$
  where $a_i$ is the <u>average dissimilarity</u> between observation $i$ and the <u>other points in the cluster to which $i$ belongs</u>. $b_i$ is the <u>average dissimilarity</u> between observation $i$ and the other points <u>in the next closest cluster to observation $i$</u>. This calculation is based on <u>rescaled data</u>.

  Observations with $s_i \approx 1$ are well-clustered. Observations with $s_i \approx 0$  mean that 2 clusters are very close, almost lie together. Observations with $s_i < 0$ are poorly clustered.

  The <u>silhouette plot</u> visualizes "cluster labels" vs. $s_i$, showing how the data is clustered.

* **What is hierarchical clustering?**

  Hierarchical clustering does not require prespecifying a particular number of cluster K. Hierarchical clustering is locally optimal (greedy), but it's very fast. We can summarize clustering in a <u>dendrogram</u>.

  There are two types of hierarchical clustering: 

  1. <u>Agglomerative clustering (bottom-up approach)</u> 

  2. <u>Divisive clustering (top-down approach)</u>

* **What is agglomerative clustering?** **What are the common approaches to measure the dissimilarity between two clusters?**

  The algorithm of agglomerative clustering is: 1) each observation starts as its own cluster, 2) at each step, two most similar clusters are combined into a new larger cluster, 3) the combination step is repeated until all observations are members of one single large cluster. Agglomerative clustering is well-suited at identifying small clusters.

  Approaches:

  * <u>Complete (or maximum) linkage clustering</u>

    Determine the <u>maximum dissimilarity</u> between any observation in the first cluster and any observation in the second cluster. (conservative)

  * <u>Single linkage clustering</u>

    Determine the <u>minimum dissimilarity</u> between any observation in the first cluster and any observation in the second cluster.

  * <u>Average linkage clustering</u>

    Compute <u>all pairwise dissimilarities</u> between observations in the first and second cluster, and calculate the <u>average</u>.

  * <u>Ward's method</u>

    Use <u>variance of observations within clusters</u> - "<u>within-cluster variation</u>". Specifically, join two clusters whose merged cluster has the smallest within-cluster sum of squared distances.

* **What are the two types of methods to choose the optimal number of clusters?**

  * <u>Direct methods</u> that involve optimizing a particular criterion (<u>elbow and silhouette methods</u>)

    These are informal approach which measures global clustering characteristics only. It is common to have inconsistent results.

    * <u>Elbow method</u>

      For the particular clustering method, let $K$ vary over a range of values. Compute total within-cluster variation $T_K$ for each K. Plot $T_K$ against $K$ and look for a clear bend ("knee") in the graph.
      $$
      T_K = \sum^K_{k=1} W(C_k)
      $$
      (When $K$ goes to infinity, $T_K$ goes to zero.)

    * <u>Average silhouette method</u>

      For the particular clustering method, let $K$ vary over a range of values. For each $K$, calculate the average silhouette across all observations. Plot $S_j$ against $K$.
      $$
      S_K = {1\over n} \sum^n_{i=1} s_i
      $$
      The value of $K$ where $S_K$ is maximized is considered the appropriate number of clusters.

      (When $K$ goes to infinity, $T_K$ will eventually become 1. )

  * <u>Testing methods</u> that evaluate evidence against a null hypothesis (<u>gap statistic</u>)

    * <u>Gap statistic</u>

      The idea is: for a particular choice of $K$ clusters, compare the total within cluster variation to the expected within-cluster variation under the assumption that the data have no obvious clustering (i.e. randomly distributed). The gap statistic detects whether the data clustered into K groups is significantly better than if they were generated at random.

      The algorithm is: 

      1) Cluster the data at varying number of total clusters $K$. Compute $T_K$ - the total within-cluster sum of squared distances. 

      2) Generate B reference datasets of size N, with simulated values of variable $j$ uniformly generated over the range of the observed variable $x_j$.  

      3) For each generated dataset $b = 1, ..., B$, perform the clustering for each $K$. Compute the total within-cluster sum of squared distances $T_K^{(b)}$. 

      4) Compute the gap statistic and evaluate it for all possible values of $K$, choose $K$ which maximizes the gap statistic.
         $$
         Gap(K) = \left({1\over B} \sum^B_{b=1}\log (T_K^{(b)})\right) - \log (T_K)
         $$

      5) Alternatively, let $\overline{w} = {1\over B} \sum^B_{b=1}\log (T_K^{(b)})$, compute the standard deviation
         $$
         sd(K) = \sqrt{{1\over B}\sum^B_{b=1}\left(\log(T_K^{(b)} )- \overline{w}\right)^2}
         $$
         Define $s_K = sd(K)\sqrt{1 + 1/B}$. Finally, choose the number of clusters as the smallest $K$ such that
         $$
         Gap(K) \geq Gap(K+1) - s_{K + 1}
         $$
  
* **Compare the two types of methods to choose the optimal number of clusters in terms of performance.**

  * Both elbow method and average silhouette method measure global clustering characteristics only, and are informal approaches
  * Gap statistic is a more principled approach to choosing cluster sizes. It turns out to result in more conservative clusterings (fewer clusters), but does not perform as reliably when the clusters overlap.

* **What is DBSCAN and how it works?**

  Density-based clustering algorithm (DBSCAN). 

  * Can find any shape of clusters. Can identify dense region of observations.
  * Identifies observations that do not belong to clusters as outliers.
  * Does not require specifying the number of clusters (like hierarchical clustering).
  * Can be used for predicting cluster membership for new data.

  DBSCAN requires two parameters to implement:

  * $\epsilon$: the radius of a neighborhood around an observation.
    * Compute kth (k = MinPts) nearest neighbor distance for each point. Plot the distance in sorted order. Look for a bend ('knee') in the plot, and use the distance at the knee as the choice.
  * MinPts: the minimum number of points within an $\epsilon$ radius of an observation to be considered a 'core' point.
    * Default 5, at least 3 to produce non-trivial clustering. 5-6 is sufficient most of the time.
    * Larger values may be better fro large data sets, for noisy data, or for data that contain repeats.

  Three types of points defined in DBSCAN algorithm:

  * <u>Core points</u>: observations with MinPts total observations within an $\epsilon$ radius.
  * <u>Border points</u>: observations that are not core points, but are within $\epsilon$ of a core point.
  * <u>Noise points</u>: everything else.

  Two terms defined in DBSCAN algorithm:

  * <u>Density-reachable</u>: point A is density reachable from point B if there is a set of core points leading from B to A. (B must be a core point, A can be a core or border point.)
    * Not symmetric: border point of pre core point is the core point of the next; if B is not a core point, then A may not be density reachable from B even if B is density reachable from A. (A and B are not necessarily core points. )
  * <u>Density-connected</u>: two points A and B are density connected if there is a core point C such that both A and B are density reachable from C.
    * A density based cluster is defined as a group of density connected points.

## 2. Bayesian Statistics

* **What is the main difference between classical statistics and bayesian statistics?**

  * The classical statistics and bayesian statistics are founded on different definition of probability.

    * <u>Frequency definition of probability</u> where classical / frequentist statistics founded on:
      * Probability of an event is its <u>long-run frequency</u> of occurrence. 
      * This definition is useful for describing the likelihood of potentially observable data.
      * Statistics founded on the Frequentist definition of probability can only assign probabilities to future data or potentially observable quantities.

    * <u>Subjective definition of probability</u> where bayesian statistics founded on:
      * Probability of an event is one's <u>degree of belief</u> that the event will occur. 
        * This definition is useful for quantifying beliefs about non-data events (but also data events).
        * Statistics founded on the subjective definition of probability can consider probabilities of values of unknown parameters (as well as values of potentially observable quantities).

    Classical and Bayesian statistics coincides much more when data is large, because when data is large, prior information becomes less important.

  * A main difference operationally between classical methods and bayesian method is the incorporation of a prior distribution.

* **What is the benefits of bayesian statistics?**

  Given a probability model, bayesian analysis makes full use of all the data. 

  Statistical inferences that are unacceptable must come from inappropriate modeling assumptions, not a problem in the underlying inferential mechanism.

  Awkward problems that Frequentists face (e.g. choice of estimators, adjustments for certain types of data) do not arise in Bayesians

  Modern computational methods (MCMC etc) enables fitting even complex data models.

* **What is Bayes Rule?**
  $$
  \begin{aligned}
  p(\theta|\bold{y}) &= {p(\theta) p(\bold{y} | \theta) \over \int p(\theta) p(\bold{y} | \theta)d\theta} = {p(\theta) p(\bold{y} | \theta) \over p(\bold{y})}\\
  & \propto p(\theta)p(\bold{y}|\theta) \propto p(\theta) L(\theta | \bold{y})
  \end{aligned}
  $$
  Posterior $\propto$ Prior $\times$ Likelihood 

* **What is the key idea of bayesian statistics?**

  In bayesian statistics, all quantities (data, unknown parameters) have probability distributions to describe uncertainty.

  * <u>Parameters</u>: there is a probability distribution $p(\theta)$ called 'priori' to describe uncertainty in model parameters. The parameter is not random but a fixed value. The <u>prior</u> distribution of the unknown parameters represents the state of knowledge prior to observing the data.
  * <u>Data</u>: observations obtained from a data-generating mechanism are assumed to follow a probability distribution $p(\bold{y}|\theta)$. This is called <u>likelihood</u> function $p(y|\theta)= L(\theta|\bold{y})$ , which is the probability of the data (probability density for continuous outcomes) conditional on the parameters, viewed as a function of the parameters.

* **How to choose prior distribution?**

  * There are two types of prior distributions:
    * <u>Informative prior distribution</u>: use knowledge and expertise to construct a prior distribution that properly reflects prior beliefs about the unknown parameters
    * <u>Non-informative prior distribution</u>: choose a distribution objectively, acting as though no prior knowledge about the parameters exists before observing the data. Non-informative prior distributions are termed 'vague', 'diffuse', and 'objective'.
  * Comparison:
    * The best and most scientific approach would be to construct a defensible informative prior distribution. However, someone would argue that informative prior distributions are not objective or scientific because different Bayesians could given different priors and end up with different posteriors.
    * It is often desirable to have prior distributions that 1) formally express ignorance and 2) can be viewed as default choices when no prior knowledge is available. A objective prior distribution could assign 'equal probability' to all values of the parameters, and so the likelihood could overwhelm the prior density with even small amounts of data. However, the problems are 1) often there is no unique non-informative prior distribution, 2) it's hard to satisfy that any method for constructing a non-informative prior distribution should be invariant to the scale of the parameter, 3) common methods for constructing non-informative prior distributions result in 'improper' probability distributions.

* **What is the main goal of bayesian inference?**

  The main goal is to 1) <u>obtain the posterior distribution of the unknown parameters</u>. It is the probability distribution of the unknown parameters conditional on the observed data. It is the probability distribution that describes the state of knowledge about the parameters once the data have been observed. 

  And then 2) <u>study the posterior distribution, or derive summaries from the posterior distribution</u>. The summary includes:

  * Posterior mean $E(\theta | y)$.

  * Posterior mode (value of $\theta$ that maximizes $p(\theta|\bold{y})$).

  * Central posterior interval for $\theta$. (interval incorporates 95% probability of $p(\theta | \bold{y})$).

    (Note: classical statistics usually assume a known distribution so the confidence interval (used by frequentist in classical statistics) is usually wider than central posterior interval)

  * Highest posterior density (HPD) region for $\theta$ - shortest interval with specified probability. 

* **What are steps of performing bayesian analysis?**

  1. Formulate a probability model for the data.

  2. Decide on a prior distribution for the unknown model parameters.

     (Note: usually improper prior distribution lead to proper posterior distributions, but one needs to check. A proper prior distribution always leads to a proper posterior distribution.)

  3. Observe the data, and construct the likelihood function based on the data.

     (Note: the likelihood function is the joint probability of the data.)

  4. Determine the posterior distribution 

     (Note: posterior = prior + likelihood, obtained by bayes rule. Usually the normalizing constant cannot be determined analytically.)

  5. Summarize important features of the posterior distribution, or calculate quantities of interest based on the posterior distribution.

* **Why bayesian models can be seen as generative models?**

  * A g<u>enerative model</u> is a probabilistic specification to generate data, which contains

    * Parameters $\theta$
    * Features (or predictors or covariates) $x$
    * Labels (or the outcome variable) $y$

  * In bayesian framework we assume

    * Prior distribution $\theta \sim p(\theta)$
    * Data probability model $y|\theta, x \sim p(y | \theta, x)$

    So we can generate or simulate outcomes by

    * Generate a value $\theta$ from $p(\theta)$
    * Given $\theta$ and known covariates $x$, simulate an outcome $y$ from $p(y | \theta ,x)$ ( $p(y | \theta ,x)$ is called <u>prior predictive distribution</u>.)

* **Why bayesian models can be seen as predictive models?**

  * One of the strength of bayesian is the ability to produce model-based predictions that accounts for the uncertainty in parameter inferences.

  * We can determine $p(\tilde{y}| \bold{y})$, the probability distribution for a new value $\tilde{y}$ given the data we already analyzed. The distribution is called <u>posterior predictive distribution</u>.

  * Note that $\theta$ is no longer in the expression - the posterior predictive distribution "averages out" the uncertainty about $\theta$ over the posterior distribution.

  * Analytic solution:
    $$
    p(\tilde{y}|\bold{y}) = \int p(\tilde{y}|\theta)p(\theta | \bold{y}) d\theta
    $$
    This can be computed via simulation:

    1. Generate 10000 simulated parameter values $\theta^{(1)}, ...\theta^{(10000)}$ from the posterior distribution.
    2. For each $j = 1, ..., 10000$ separately, generate a value of $\tilde{y}^{(j)}$ from the probability model with parameter value $\theta^{(j)}$.
    3. Summarize features (mean, etc) of the 10000 values $\tilde{y}^{(1)}, ...,\tilde{y}^{(10000)}$ for predictions.

* **What are the challenges of Bayesian approach in more complex modeling situation?**

  * The posterior density for most real-life models can be complex expressions.
  * The posterior density cannot be written exactly because the normalizing constant cannot be evaluated.
  * Even we can write down the posterior density exactly, it is difficult to summarize using standard analytic tools.

* **Why we use Bayes theorem instead of conditional probability?**

  Conditional probability is the likelihood of an outcome occurring, based on a previous outcome occurring. Bayes' theorem provides a way to revise existing predictions or theories (update probabilities) given new or additional evidence.

* **What are the approaches used to summarize posteriors?**

  * Quadrature methods
  * (Direct) Monte Carlo (MC) simulation
  * Indirect MC methods: 
    * Rejection sampling
    * Weighted bootstrap
  * Monte Carlo Markov Chain (MCMC) simulation
    * Random Walk Metropolis (RWM) sampling
    * Gibbs sampling

* **What is quadrature method and how it works to summarize posterior?**

  * The basic idea is to replace $p(\theta| \bold{y})$ with an approximating discrete mass function $\tilde{p}(\theta | \bold{y})$, obtain posterior summaries through the approximating discrete distribution. 
  * A common version is <u>Gauss-Hermite Quadrature</u>. 
    * Choose the spacing of the values of $\theta$ according to a normal distribution along with relevant weights.
    * Requires computation to approximate the normal mean and variance.
    * For multivariate parameter $\theta$, the quadrature spacing is typically accomplished one variable at a time.
  * Problems results in limited applicability:
    *  The quadrature grid of values of $\theta$ may not adequately represent the true distribution of $\theta$.
    * With multivariate $\theta$, it's very difficult to obtain reasonable representation of discrete values in multivariate space (curse of dimensionality) without making the computation unwieldly.

* **What is Monte Carlo simulation and how it works to summarize posterior distribution?**

  * MC simulation is valid to summarize posterior distribution because <u>summarizing computer-simulated values from the posterior distribution is a legitimate alternative to analytic summaries</u>. 
  * Calculate the sample-version of the distribution from the simulated sample. Usually need to simulate a very large sample in order to precisely approximate the generating distribution.
  * Steps:
    1. Simulate 10000 values from the posterior distribution
    2. Report sample summaries from the distribution of 10000 simulated values

* **What are Indirect Monte Carlo methods and how they work?**

  Indirect MC methods include rejection sampling and weighted bootstrap.

  * <u>Rejection sampling</u>:

    * Suppose we are going to simulate values from density $h(\theta)$ but cannot do it directly. $h(\theta)$ can be unnormalized $\int h(\theta) d\theta = c \neq 1$. Suppose we can simulate directly from density $g(\theta)$. We need to ensure that there exists constant $M > 0$ such that $h(\theta) / g(\theta) \leq M ~~\forall \theta$.

    * To obtain a simulated $\theta$ from $h(\theta)$, do
      1. Simulate $\theta$ from $g(\theta)$.
      2. Simulate $U$ from a uniform distribution on $(0,1)$.
      3. If $U \leq h(\theta)/ Mg(\theta)$, then accept $\theta$. Otherwise, repeat steps 1-3.
    * Most efficient way of choosing $M$: choose smallest $M$ such that the normal curve $Mg(\theta)$ stays above the $h(\theta)$. Because when $Mg(\theta)$ is too higher than $h(\theta)$, probability of keeping candidate $\theta$ is lower so the method will be inefficient - we waste computation and time to reject a lot of $\theta$.

  * <u>Weighted bootstrap</u>

    Suppose we cannot readily produce an $M$ such that $h(\theta) / g(\theta) \leq M ~~ \forall \theta$. We can do:

    1. Simulate $K$ draws $\{\theta_1, ..., \theta_K\}$ from $g(\theta)$.
    2. For each $k=1, ..., K$, compute $w_k = h(\theta_k)/ g(\theta_k)$, and let $q_k = w_k / \sum^K_{j=1}w_j$, the normalized values of the $w_k$.
    3. Now simulate $\theta$ from the discrete distribution of values $\{\theta_1, ..., \theta_K\}$ with probabilities $\{q_1, ..., q_K\}$.

    The resulting $\theta$ is approximately distributed according to $h(\theta)$, with approximation improving as $K$ increases. This procedure aka <u>sampling / importance resampling (SIR)</u>

* **How Indirect Monte Carlo methods work in summarizing posterior?**

  * <u>Bayesian rejection sampling</u>

    Suppose we have prior $p(\theta)$, likelihood $L(\theta | y)$, and posterior $p(\theta | \bold{y})$. Let $h(\bold{\theta}) = p(\bold{\theta}) L(\bold{\theta} | \bold{y})$ (a constant factor of the posterior density), and $g(\bold{\theta}) = p(\bold{\theta})$. Let $M$ be the smallest value for which $h(\theta) / g(\theta) \leq M~~ \forall \theta$. i.e.
    $$
    {h(\theta)\over g(\theta)}= {p(\theta) L(\theta | \bold{y})\over p(\theta)} = L(\theta | \bold{y}) \leq M
    $$
    Thus choose $M = L(\widehat{\theta}|\bold{y})$ where $\widehat{\theta}$ is the MLE. Then do

    1. Simulate $\theta$ from $p(\theta)$.
    2. Generate $U$ from a uniform distribution over $(0,1)$.
    3. If $U \leq h(\theta) / Mg(\theta) = L(\theta | \bold{y}) / L(\widehat{\theta}| \bold{y})$, then accept $\theta$ as a simulated value from $p(\theta | \bold{y})$, otherwise repeat.

  * <u>Bayesian weighted bootstrap</u>

    Let $h(\bold{\theta}) = p(\bold{\theta}) L(\bold{\theta} | \bold{y})$, and $g(\bold{\theta}) = p(\bold{\theta})$. 

    1. Simulate $\{\theta_1, ..., \theta_K\}$ from $p(\theta)$.
    2. For each $k=1, ..., K$, compute $w_k = h(\theta_k)/ g(\theta_k) = L(\theta_k | \bold{y})$, and let $q_k = w_k / \sum^K_{j=1}w_j = L(\theta_k | \bold{y})/ \sum^K_{i=1} L(\theta_j | \bold{y})$, the normalized values of the $w_k$.
    3. Now simulate $\theta$ from the discrete distribution of values $\{\theta_1, ..., \theta_K\}$ with probabilities $\{q_1, ..., q_K\}$.

* **What are the problems of Indirect Monte Carlo methods?**

  * The problem is they are <u>computational inefficient</u>. 

    * They work well when $h(\theta) / g(\theta) = L(\theta | \bold{y})$ is nearly constant with respect to $\theta$. 

    * However, in real world, almost all posterior $p(\theta | \bold{y})$ is peak (because when data is large, the likelihood is peak). So with the indirect MC methods, almost all $\theta$ are from the peak point.

    * When $h(\theta) / g(\theta) = L(\theta | \bold{y})$ is not constant, then
      * It could take ages in rejection sampling before a $\theta$ is accepted.
        
        The probabilities $\{q_1, ..., q_K\}$ in the weighted bootstrap could be terribly skewed so that the discrete approximation to $p(\theta | \bold{y})$ is horrible.

  * Another problem is these methods require proper prior. If we assume an improper prior distribution, we cannot simulate from $p(\theta)$.

* **What is Monte Carlo Markov Chain (MCMC) simulation and how it works?**

  The key idea is to <u>create a Markov chain whose stationary distribution is $p(\theta | \bold{y})$</u>. Values are computer-simulated from the Markov Chain.
  $$
  \theta^k \sim P(\theta | y, \theta^{k-1}), \theta^k \xrightarrow[]{k \rightarrow \infty} p(\theta | \bold{y})
  $$
  There are two ways to construct a Markov Chain:

  * <u>Random Walk Metropolis (RWM) sampling</u>

    Idea: We want to sample vectors of $\theta$ from posterior density of d-dimensional $\theta$: $p(\theta | \bold{y})$.

    Steps:

    1. Pick arbitrary starting vector $\theta_1$

    2. For $j = 1,2,..$

       a. Simulate d-dimensional vector $\Delta_j$ (the proposed 'jump') from pre-specified distribution $\tilde{p}(\Delta)$. ($\tilde{p}(\Delta)$ is usually bell-shaped, equal possibility of directions, can be tuned)

       b. Evaluate the proposal acceptance probability
       $$
       \alpha(\theta_j, \Delta_j) = \min \left(1, {p(\theta_j + \Delta_j | \bold{y})\over p(\theta_j | \bold{y})}\right)
       $$
       c. let
       $$
       \theta_{j+1} = \begin{cases} \theta_j, &\text{with probability } 1-\alpha(\theta_j, \Delta_j) \\
       \theta_j + \Delta_j, & \text{with probability } \alpha(\theta_j, \Delta_j)\end{cases}
       $$

    Interpretation:

    * If uphill proposals (goes to local maximum) are always accepted. Intuitively, if $p(\theta_j + \Delta_j | \bold{y}) \geq p(\theta_j | \bold{y})$, $p(\text{move}) =1$. 

    * If downhill proposals (move away from a local maximum) are accepted with probability equal to the relative heights of the posterior density at the proposed and current values. Intuitively, if $p(\theta_j + \Delta_j | \bold{y}) < p(\theta_j | \bold{y}), p(\text{move}) = \alpha$.
    * As $j \rightarrow \infty$, $\theta_j$ is simulated value from $p(\theta | \bold{y})$.

    Drawback:

    * The direction of the proposed jump is unrelated to the location of the highest posterior probability, so it is inefficient in that it proposes jumps at random when seeking local maximum.

  * <u>Gibbs sampling</u>

    Idea: $c p(\theta_1, ..., \theta_d | \bold{y}) = p(\theta_1 | \theta_2, ..., \theta_d, \bold{y})$, where $c$ is a constant factor and $\theta_i$ are dependent.

    Steps:

    1. Select arbitrary starting parameter values $\theta_2^{(1)},\theta_3^{(1)},...,\theta_d^{(1)}$.
    2. For iteration $j = 2,3,...$,
       * Simulate $\theta_1^{j} \sim p(\theta_1|\theta_2^{(j-1)},...,\theta_d^{(j-1)}, \bold{y})$
       * Simulate $\theta_2^{j} \sim p(\theta_2|\theta_1^{(j-1)},...,\theta_d^{(j-1)}, \bold{y})$
       * ...
       * Simulate $\theta_d^{j} \sim p(\theta_d|\theta_1^{(j-1)},...,\theta_{d-1}^{(j-1)}, \bold{y})$

    Interpretation:

    * In step 2, we simulate from the conditional posterior distribution of each $\theta_k$ given the data and all of the other parameters at their current values.

    * As $j \rightarrow \infty, \theta_j = (\theta_1^{(j)},...,\theta_d^{(j)})$ is a simulated value from $p(\theta | \bold{y})$.

    Similarity with RWM:

    * Gibbs sampler is a special case of RWM.
    * A proposed parameter is simulated (the 'jump')
    * The proposal is accepted with a specified probability

    Difference from RWM:

    * In Gibbs sampler, the proposal distribution is more 'guided' by the data and other parameter values than RWM sampling
    * In Gibbs sampler, the acceptance probability happens to always be 1.
    * Gibbs sampler turns simulating from a multi-parameter distribution into simulating from a sequence of one-parameter distributions, which is much more tractable.

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

* **How to diagnose the convergence of MCMC simulation?**

  * <u>Visualization</u>: run parallel MCMC with different starting points, after burn-in period, plot the parameter draws as a function of iteration for each parameter, look for whether the chains coalesce.

  * <u>R-hat statistic</u>: a numerical measure indicating whether the Markov Chain is running long enough to reach convergence. Specifically, $\hat{R}$ is a metric for comparing how well a chain has converged to the equilibrium distribution by comparing its behavior to other randomly initialized Markov chains. Multiple chains initialized from different initial conditions should give similar results. $\hat{R}$ is a necessary but not sufficient condition.
    $$
    \text{R-hat} = {\text{between-chain variance}\over \text{within-chain variance}}
    $$

    * $\sim 1.0$ indicates convergence - all chains converge to the same equilibrium.
    * Large values ($>1.3$) indicates not long run enough or the parameter  may be difficult to obtain reasonable samples given strong autocorrelation in the sampler.

* **What is the difference between direct/indirect MC simulation and MCMC simulation?**

  Direct / indirect MC simulation simulates from $p(\theta | \bold{y})$.

  MCMC simulation simulates a random walk in the space of $\theta$ which converges to $p(\theta | \bold{y})$.

* **What is Hierarchical modeling in Bayesian framework?**

  Hierarchical modeling is a compromise between separate regressions and one overall regression.

  For each group there is a $\beta_g \sim \mathcal{N}(\mu_\beta, \Sigma_\beta)$. $\mathcal{N}(\mu_\beta, \Sigma_\beta)$ is called normal 'random effects' distribution. 

  * The data across all groups inform the values of $\mu_\beta, \Sigma_\beta$, meaning that the $\beta_g$ are informed from other groups besides the data in group $g$. Sometimes called '<u>partial pooling</u>' of data across groups. 
  * They can vary but the random effects distribution keeps them from being too far apart. 
  * The random effects distribution 'shrinks' the $\beta_g$ to a common population mean. Shrinkage tends to be particularly noticeable when some groups have small numbers of observations relative to others.
  * Assuming a population distribution on the $\beta_g$ with unknown covariance $\Sigma_\beta$ can be viewed as a form of <u>regularization</u>.

* **Briefly introduce a popular Bayesian analysis package**

  PyMC3 is a Python library for programming Bayesian analysis, and more specifically, data creation, model definition, model fitting, and posterior analysis. It uses the concept of a `model` which contains assigned parametric statistical distributions to unknown quantities in the model. Within models we define random variables and their distributions.

  PyMC3 uses the <u>No-U-Turn Sampler (NUTS)</u> and the <u>Random Walk Metropolis</u>, two Markov chain Monte Carlo (MCMC) algorithms for sampling in posterior space. Monte Carlo gets into the name because when we sample in posterior space, we choose our next move via a pseudo-random process. NUTS is a sophisticated algorithm that can handle a large number of unknown (albeit continuous) variables.

