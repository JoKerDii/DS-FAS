# Hyperparameter Tuning

## Hyperparameter optimization

* Hyperparameter tuning in general

  * General pipeline
    1. Select the most influential parameters
    2. Understand, how exactly they influence the training
    3. Tune them
       1. Manually (change and examine)
       2. Automatically (hyperopt, etc)
  * Manual and automatic tuning
  * What should we understand
    * Whether model is underfitting, good fit, overfitting

* Model, libraries and hyperparameter optimization

  * Tree-based models

    * GBDT: XGBoost, LightGBM, CatBoost

      | XGBoost                             | LightGBM                               | Recommend                                                  |
      | ----------------------------------- | -------------------------------------- | ---------------------------------------------------------- |
      | max_depth                           | max_depth/num_leaves                   | Start from 7                                               |
      | subsample                           | bagging_fraction                       | Less overfitting (regularization), fit slow                |
      | colsample_bytree, colsample_bylevel | feature_fraction                       | Regularizatioin                                            |
      | Min_child_weight, lambda, alpha     | Min_data_in_leaf, lambda_l1, lambda_l2 | Regularization                                             |
      | Eta, num_round                      | Learning_rate, num_iterations          |                                                            |
      | seed                                | *_seed                                 | Make sure random seed does not affect training result much |

    * RandomForest / ExtraTrees

      Build independent trees, meaning that building more trees will not overfit the data, as oppose to XGBoost.

      n_estimators: the higher the better

      Max_depth: can be set to None, meaning unlimited depth. (10,20,higher)

      Max_features: the more the faster. 

      Min_samples_leaf

      Criterion: Gini better more often than entropy.

      Random_state

      n_jobs: number of cores.

  * Neural networks
    * PyTorch, Tensorflow, Keras
      * Number of neurons per payer
      * Number of layers
      * optimizers
        * SGD + momentum (converge slower but generalize better)
        * Adam/ adadelta/ adagrad/... (in practice lead to more overfitting)
      * Batch size: larger -> more overfitting (32, 64)
      * Learning rate (start from large lr like 0.1) 
      * Regularization 
        * L2/l1 for weights
        * Dropout / dropconnect (do not add dropout at the first layer)
        * Static dropconnect

  * Linear models
    * SVM/ SVR
      * Sklearn wraps libLibear and libSVM
      * Compile yourself for multicore support
    * Logistic regression
      * LogisticRegression / LinearRegression + regularizers
    * SGDClassifier / SGDRegressor
    * Vowpal Wabbit
      * FTRL

## Resources

* Sklearn Grid Search [[link](https://scikit-learn.org/stable/modules/grid_search.html)]
* Complete Machine Learning Guide to Parameter Tuning in Gradient Boosting (GBM) in Python [[link](https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/)]