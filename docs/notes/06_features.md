# Advanced Features

Feature engineering is the process of transforming raw data into ‘features’ that better represent the underlying structure and patterns in the data, so that your predictive models can be more accurate when predicting unseen data

## Statistics and distance based features

* Example of group by

  ```python
  gb = df.groupby(['user_id', 'page_id'], as_index = False).agg({'ad_price':{'max_price':np.max, 'min_price': np.min}})
  gb.columns = ['user_id', 'page_id', 'min_price', 'max_price']
  
  df = pd.merge(df, bg, how = 'left', on = ['user_id', 'page_id'])
  ```

## Matrix factorizations

* Example of feature fusion

  Vanilla BOW  + (BOW+TFIDF) + BOW(bigrams) => dimensionality reduction => tree-based models.

* Implement MF

  * Standard tools: SVD and PCA
  * For sparse matrices: TruncatedSVD
  * For count data (non negative): non-negative matrix factorization (NMF)

## Feature Interactions

* Frequent operations
  * Multiplication, sum, diff, division
* It's difficult for tree-based models to extract such dependencies, this is why FI are very important for tree-based methods.
* To limit number of features:
  * Dimensionality reduction
  * Feature selection
* Example of interaction generation pipeline
  1. Fit random forests
  2. Get feature importance
  3. Select a few most important features

## tSNE

Non-linear method of dimensionality reduction.

## Resources

* Feature transformation with ensemble of trees. [[link](https://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html)]