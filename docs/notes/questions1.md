# Machine Learning Concept Q&A

## 1. Unsupervised Learning & Cluster Analysis

* **What is unsupervised learning? Give some examples.**

  Unsupervised learning aims to discover subgroups among variables or  observations, but not about how variables or observations relate to a response variable. It can be used as a descriptive tool.

  Examples include: to find subgroups of breast cancer patients according to similar gene expressions; to find subgroups among online shoppers according to their browsing and purchasing patterns; to find subgroups of movies according to ratings assigned by movie viewers.

* **What is the difference between PCA and clustering?**

  <u>PCA</u> attempts to find a low-dimensional representation of the observations that explains a large fraction of the variation, often used for dimensionality reduction, also be useful for visualization of grouping of data.

  <u>Clustering</u> aims to find homogeneous subgroups among the observations.

* **What is clustering? **

  Cluster analysis aims to find subgroups or clusters in a multivariate dataset. It partitions the data into distinct groups so that observations within each group are similar to each other.  It is usually domain-specific to concretely define what it means for observations to be similar or different.

  Clustering involves two steps 1) calculating inter-observation distances 2) applying clustering algorithm.

* **What are the two main types of approaches to clustering?**

    Two main types of approaches to clustering are

  1. partitioning clustering

     Partitioning clustering specifies the number of clusters in advance, and then invoke an algorithm to partition the data. 

  2. hierarchical clustering

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

  Silhouette plot is a diagnostic method for any clustering including k-means. Once a clustering has be determined, we can calculate <u>silhouette</u> for observation $i$ as:
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
         