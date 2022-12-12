# Data Preprocessing

Features: numeric, categorical, ordinal, datetime, coordinates

Missing values.

## 1. Numeric features

* Preprocessing

  * Tree-based models

    Decision tree does not depend on the scaling of variables

  * Non-tree-based models

    KNN, linear model (regularization), neural networks (gradient descent) are sensitive to scaling

  * Scaling Methods:

    1. To [0,1]: `sklearn.preprocessing.MinMaxScaler`

    2. To mean=0, std = 1: `sklearn.preprocessing.StandardScalers`

    3. Preprocessing outliers

       Clip by an upper bound and a lower bound - "winsorization".

       ```python
       upper, lower = np.percentile(x, [1,99])
       y = np.clip(x, upper, lower)
       pd.Series(y).hist(bins = 30)
       ```

    4. Preprocessing ranks

       Make the outlier closer to the other objects

       ```python
       # Example
       rank([-100, 0, 1e5]) = [0,1,2]
       rank([1000,1,10]) = [2,0,1]
       ```

       ```python
       scipy.stats.rankdata
       ```

    5. Log transformation `np.log(1+x)`

    6. Raising to the power $<1$, `np.sqrt(x+2/3)`

* Feature generation 

  * Key: creativity and data understanding
  * Power: prior knowledge and EDA
  * Example:
    * Add multiplication, divisions
    * Fractional part of the price

  > **Question 1**: Suppose we have a feature with all the values between 0 and 1 except few outliers larger than 1. What can help us to decrease outliers' influence on non-tree models?
  >
  > * <u>Apply rank transform to the features</u>. Yes, because after applying rank distance between all adjacent objects in a sorted array is 1, outliers now will be very close to other samples.
  > * <u>Apply np.log1p(x) transform to the data</u>. This transformation is non-linear and will move outliers relatively closer to other samples.
  > * <u>Apply np.sqrt(x) transform to the data</u>. This transformation is non-linear and will move outliers relatively closer to other samples.
  > * <u>Winsorization</u>. The main purpose of winsorization is to remove outliers by clipping feature's values.

## 2. Categorical and ordinal features

* Preprocessing

  * Label encoding

    In alphabetical or sorted order: `sklearn.preprocessing.LabelEncoder`

    In the order of appearance: `pandas.factorize`

  * Map values to their frequencies

    ```python
    # [S,C,Q] = [0.5, 0.3, 0.2]
    encoding = titanic.groupby('Embarked').size()
    encoding = encoding/len(titanic)
    titanic['enc'] = titanic.Embarked.map(encoding)
    ```
    
  * One-hot encoding

    `pandas.get_dummies`

    `sklearn.preprocessing.OneHotEncoder`

* Feature generation

  * Combine categorical features

  > **Question 2**: Suppose we fit a tree-based model. In which cases label encoding can be better to use than one-hot encoding?
  >
  > * <u>When categorical feature is ordinal</u>. Correct! Label encoding can lead to better quality if it preserves correct order of values. In this case a split made by a tree will divide the feature to values 'lower' and 'higher' that the value chosen for this split.
  > * <u>When we can come up with label encoder, that assigns close labels to similar (in terms of target) categories.</u> Correct! First, in this case tree will achieve the same quality with less amount of splits, and second, this encoding will help to treat rare categories.
  > * <u>When the number of categorical features in the dataset is huge</u>. One-hot encoding a categorical feature with huge number of values can lead to (1) high memory consumption and (2) the case when non-categorical features are rarely used by model. You can deal with the 1st case if you employ sparse matrices. The 2nd case can occur if you build a tree using only a subset of features. For example, if you have 9 numeric features and 1 categorical with 100 unique values and you one-hot-encoded that categorical feature, you will get 109 features. If a tree is built with only a subset of features, initial 9 numeric features will rarely be used. In this case, you can increase the parameter controlling size of this subset. In xgboost it is called *colsample_bytree,* in sklearn's Random Forest *max_features.*
  >
  > **Question 3**: Suppose we fit a tree-based model on several categorical features. In which cases applying one-hot encoding can be better to use than label-encoding?
  >
  > * <u>If target dependence on the label encoded feature is very non-linear, i.e. values that are close to each other in the label encode feature correspond to target values that aren't close</u>. Correct! If this feature is important, a tree would try to make a lot of splits and select each feature' value in a category on its own. But because tree is build in a greedy way, it can be hard to select one important value in label encoded vector. This won't be the problem if you use OHE.

## 3. Datetime

* Periodicity

  Number in day, week, month, season, year, second, minute, hour

* Time since

  * Row independent moment

    e.g. number of years since 1 Jan 1970

  * Row dependent important moment

    Number of days left until next holidays / time passed after last holiday

    e.g. split daytime to date, weekday, day number since year 2014, whether is holiday, days till holidays

* Difference between dates

  * e.g. In churn prediction: use 'last purchase data' and 'last call date' and add number of days between these two events (`day_diff`)

## 4. Coordinates

* Extract interesting points on the map from training data
  * Divide the map by grid/ square, within each square, find the most expensive flat, for every other object in this square, add the distance to that flat
  * Organize the data into clusters and use center of clusters as such important points
  * Find some special areas, like the area with very old buildings and add distance to this one
  * Calculate aggregated statistics for object surrounding area
    * Number of flat around this area: area popularity
    * Mean realty price: how expensive area around selected point is 

## 5. Missing data

*  Reason: 

   - Many ML algo failed to perform on data if it contains missing values. Some may work with missing values: kNN, XGBoost, and Naïve Bayes.
   - You may end up building biased model that leads to incorrect results (less accuracy or precision)

*  Methods to handle missing values:

   - Figure out why missing

   - Drop

     - Complete Case Analysis (CCA)

       Discarding observations that contain missing values. If Data MCAT.

     - If missing values are MAR or MCAR 

     - Smaller sample size -> less predictive power

     - Deleting rows or columns

   - Impute 
   
     - Fill NA with mean, median, mode, etc
     - Random sample imputation: replace all missing data by the values random sampled from the data
     - Multiple imputation (MICE)
     - kNN imputer
     - Predicting missing data by regression on other variables 

   - Can be part of feature engineering

     Add indicator variable indicating whether the value is missing
   
   - XGBoost, Naïve Bayes can handle NaN without imputation preprocessing
   

## 6. Feature extraction from text

* Pipeline of applying Bag of Words (BOW)

  1. Texts preprocessing

     1. Lowercase

     2. Lemmatization

     3. Stemming

     4. Stopwords

        * Articles or prepositions
        * Very common words

        NLTK, natural language toolkit library for python

        `sklearn.feature_extraction.text.CountVectorizer` (max_df)


  2. Bag of words: 

     Count the number of occurrences: 

     `sklearn.feature_extraction.text.CountVectorizer`

  3. Term frequency "TFiDF":

     ```python
     tf = 1/x.sum(axis = 1)[:,None]
     x = x * tf
     ```

     Inverse document frequency

     ```python
     idf = np.log(x.shape[0] / (x>0).sum(0))
     x = x * idf
     ```

     `sklearn.feature_extraction.text.TfidfVectorizer`


  4. N-grams

     Help use local context around each word. N-grams features are typically sparse

     `sklearn.feature_extraction.text.CountVectorizer` (Ngram_range, analyzer)

  5. Embeddings (word2vec)

     Pre-trained models:

     * Words: Word2vec, Glove, FastText, etc
     * Sentences: Doc2vec, etc

  * Compare BOW and w2v
    1. Bag of Works
       * Very large vectors
       * Meaning of each value in vector is know
    2. Word2vec
       * Relatively small vectors
       * Values in vector can be interpreted only in some cases
       * The words with similar meaning often have similar embeddings


## 7. Feature extraction from images

1. Descriptors

   Features can be extracted from different layers

2. Train network from scratch

3. Fine-tuning pre-trained models

   * Allow tune all parameters which extracts more effective image representations
   * Better than train network from scratch if we have too little data
   * Lead to better results and faster training procedure

4. Augmentation

## References

* Preprocessing data - sklearn [[link](https://scikit-learn.org/stable/modules/preprocessing.html)]

  Scaling:

  * StandardScaler: to standard normal distribution

  * MinMaxScaler: to [0,1]
  * MaxAbsScaler: to [-1,1], specially for sparse data
  * RobustScaler: specially for data with outliers; cannot be fitted to sparse inputs
  * KernelCenterer (not often used?)

  Non-linear transformation

  * QuantileTransformer: map to uniform distribution (0,1)
  * PowerTransformer: map to Gaussian distribution

  Normalization

  * normalize

  Encoding categorical features

  * OrdinalEncoder
  * OneHotEncoder

  Discretization (quantization or binning)

  * KBinsDiscretizer: discretize features into k bins
  * Binarizer: thresholding numerical features to get boolean values

  Imputation

  * SimpleImputer: univariate imputation
  * IterativeImputer: multivariate imputation
  * KNNImputer: need standard scaling before imputation

* Feature scaling and normalization for ML - Sebastian raschka blog [[link](https://sebastianraschka.com/Articles/2014_about_feature_scaling.html)]

* Feature engineering - machine learning mastery [[link](https://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/)]

* Text feature extraction - blog [[link](https://andhint.github.io/machine-learning/nlp/Feature-Extraction-From-Text/)]

* Word2vec tutorial [[link](https://www.tensorflow.org/tutorials/text/word2vec), [link](https://rare-technologies.com/word2vec-tutorial/), [link](http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/)]

