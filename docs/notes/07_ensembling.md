# Ensembling

## Ensemble Methods

* Averaging (or blending)

* Weighted averaging

* Conditional averaging

* Bagging

* Boosting

  * Weight based 

    AdaBoost

  * Residual based

    XGBoost, LightGBM, H2O GBM, CatBoost

* Stacking

  ```python
  training, valid, ytraining, yvalid = train_test_split(train, y, text_size = 0.5)
  
  model1 = RandomForestRegressor()
  model2 = LinearRegression()
  
  model1.fit(training, ytraining)
  model2.fit(training, ytraining)
  
  preds1 = model1.predict(valid)
  preds2 = model2.predict(valid)
  
  test_preds1 = model1.predict(test)
  test_preds2 = model2.predict(test)
  
  stacked_predictions = np.column_stack((pred1, pred2))
  stacked_test_predictions = np.column_stack((test_preds1, test_preds2))
  
  meta_model.fit(stacked_predictions, yvalid)
  
  final_predictions = meta_model.predict(stacked_test_predictions)
  ```

  * Software
    * [StackNet](https://github.com/kaz-Anova/StackNet)
    * Stacked ensembles from H2O
    * [Xcessiv](https://github.com/reiinakano/xcessiv)

* CatBoost

## 5 Validation Schemes for 2nd Level Models (Stacking)

**a. Simple holdout scheme**

1. Split train data into three parts: partA and partB and partC.
2. Fit N diverse **models** on partA, predict for partB, partC, test_data getting *meta-features* partB_meta, partC_meta and test_meta respectively.
3. Fit a **metamodel** to a partB_meta while validating its hyperparameters on partC_meta.
4. When the **metamodel** is validated, fit it to [partB_meta, partC_meta] and predict for test_meta.

**b. Meta holdout scheme with OOF meta-features**

1. Split train data into K folds. Iterate though each fold: retrain N diverse **models** on all folds except current fold, predict for the current fold. After this step for each object in train_data we will have N *meta-features* (also known as *out-of-fold predictions, OOF*). Let's call them train_meta. 
2. Fit **models** to whole train data and predict for test data. Let's call these features test_meta.
3. Split train_meta into two parts: train_metaA and train_metaB. Fit a **meta-model** to train_metaA while validating its hyperparameters on train_metaB.
4. When the **meta-model** is validated, fit it to  train_meta and predict for test_meta.

**c. Meta KFold scheme with OOF meta-features**

1. Obtain *OOF predictions* train_meta and test metafeatures test_meta using **b.1** and **b.2.**
2. Use KFold scheme on train_meta to validate hyperparameters for **meta-model**. A common practice to fix seed for this KFold to be the same as seed for KFold used to get *OOF predictions*. 
3. When the **meta-model** is validated, fit it to train_meta and predict for test_meta.

**d. Holdout scheme with OOF meta-features**

1. Split train data into two parts: partA and partB.
2. Split partA into K folds. Iterate though each fold: retrain N diverse **models** on all folds except current fold, predict for the current fold. After this step for each object in partA we will have N *meta-features* (also known as *out-of-fold predictions, OOF*). Let's call them partA_meta.
3. Fit **models** to whole partA and predict for partB and test_data, getting partB_meta and test_meta respectively.
4. Fit a **meta-model** to a partA_meta, using partB_meta to validate its hyperparameters.
5. When the **meta-model** is validated basically do 2. and 3. without dividing train_data into parts and then train a **meta-model**. That is, first get *out-of-fold predictions* train_meta for the train_data using **models.** Then train **models** on train_data, predict for test_data, getting  test_meta. Train **meta-model** on the train_meta and predict for test_meta.

**e. KFold scheme with OOF meta-features**

1. To validate the model we basically do **d.1 -- d.4** but we divide train data into parts partA and partB M times using KFold strategy with M folds.
2. When the meta-model is validated do **d.5.**

## Tips and Tricks

* Diversity based on algorithms

  * 2-3 gradient boosted trees (lightgb, xgboost, H2O, catboost)

    [try use ones with bigger depth, middle depth and low depth, and then tune hyperparameters to make them perform roughly the same]

  * 2-3 Neural nets (keras, pytorch)

    [try use ones with more hidden layers, medium number of hidden layers, and one hidden layers]

  * 1-2 ExtraTrees / Random Forest (sklearn)

  * 1-2 linear models as in logistic / ridge regression, linear SVM (sklearn)

  * 1-2 knn models (sklearn)

  * 1 factorization machine (libfm)

  * 1 SVM with nonlinear kernel if size/ memory allows (sklearn)

* Diversity based on input data

  * Categorical features: onehot, label encoding, target encoding
  * Numerical features: outliers, binning, derivatives, percentiles
  * Interactions: column (+-*/) operations, groupby, unsupervised clustering (kmeans, pca,..)

* Simpler (or shallower) algorithms

  * Gradient boosted trees with small depth
  * Linear models with high regularization
  * Extra Trees
  * Shallow networks (1 hidden layer)
  * Knn with BrayCurtis Distance
  * Brute forcing a search for best linear weights based on cross-validation

* Difference feature engineering

  * Pairwise differences between meta features
  * Row-wise statistics like averages or stds
  * Standard feature selection techniques

* For every 7.5 models in previous level we add 1 in meta

## Resources

* [Kaggle Ensembling Guide Article on MLWave](https://usermanual.wiki/Document/Kaggle20ensembling20guide.685545114/view)
* [Heamy — a set of useful tools for competitive data science (including ensembling)](https://github.com/rushter/heamy)
* [StackNet — a computational, scalable and analytical meta modelling framework (by KazAnova)](https://github.com/kaz-Anova/StackNet)