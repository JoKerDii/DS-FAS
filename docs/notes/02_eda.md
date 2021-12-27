# EDA and Validation

## Exploratory Data Analysis (EDA)

* Tools for individual feature exploration

  * `plt.hist(x)`
  * `plt.plot(x,'.')`
  * `plt.scatter(range(len(x)),x, c=y)`
  * `df.describe()`
  * `x.value_counts()`
  * `x.isnull()`

* Tools for feature relations

  * `plt.scatter(x1,x2)`
  * `pd.scatter_matrix(df)`
  * `df.corr(), plt.matshow(...)`

* Dataset cleaning

  * Constant features

    All records have same value for a feature, check: `traintest.nunique(axis=1) == 1`, and remove this feature.

  * Duplicated numeric features

    Two features are the same for every record, check: `traintest.T.drop_duplicates()`

  * Duplicated categorical features

    They can be identical but their levels have different names, <u>label encode</u> all categorical features and compare them as if they were numbers

    ```python
    for f in categorical_feats:
      traintest[f] = traintest[f].factorize()
      
    traintest.T.drop_duplicates()
    ```

  * Duplicated rows

    Check if same rows have same label

    Find duplicated rows, understand why (mistakes?)

  * Whether the dataset is shuffled (data leakage?)

* Potential features

  * Cumulative features

    When one feature is always larger than the other feature, we can subtract one by the other and create a new feature.

    Linear models, neural networks can do this by themselves, but tree-based model couldn't.

## Validation and Overfitting

* Validation tools:

  * Holdout: ngroups = 1

    `sklearn.model_selection.ShuffleSplit`

  * K-fold: ngroups = k

    1. Split train data into K folds. 
    2. Iterate though each fold: retrain the model on all folds except current fold, predict for the current fold.
    3. Use the predictions to calculate quality on each fold. Find such hyper-parameters, that quality on each fold is maximized. 

    `sklearn.model_selection.Kfold`

  * Leave-one-out: ngroups = len(train)

    `sklearn.model_selection.LeaveOneOut`

  Note that the validation schemes are supposed to be used to estimate quality of the model. When we find the right hyper-parameters and want to get test predictions we have to retrain our model using all training data.

* Stratification:

  Stratification preserves the same target distribution over different folds.

  Useful for small and imbalanced dataset.

* Splitting data

  * Random, rowwise

  * Timewise

    Moving window validation.

    <u>e.g. "Rossman Store Sales", "Grupo Bimbo Inventory Demand"</u>

  * By id

    <u>e.g. "Intel & MobileODT Cervical Cancer Screening", "The Nature Conservancy fisheries monitoring competition"</u>

  * Combined

    <u>e.g. "Western Australia Rental Prices competition by Deloitte" and "qualification phase of data science game 2017".</u>

  Note that: Our validation should always mimic train/test split made by organizers.

* Validation problems

  * Validation stage

    Cause of different scores and optimal parameters

    1. Too little data
    2. Too diverse and inconsistent data

    We should do extensive validation

    1. Average scores from different KFold splits
    2. Tune model on one split, evaluate score on the other

    <u>e.g. "Liberty Mutual Group Property Inspection Prediction competition" and "Santander Customer Satisfaction competition".</u>

  * Submission stage

    * LB score is consistently higher/lower than validation score

      <u>e.g. "Quora Question Pairs".</u>

    * LB score is not correlated with validation score at all.

      <u>e.g. "Data Science Game 2017 Qualification phase: Music recommendation", "CTR prediction task from EDA"</u>

    Reasons for problems: 

    * Too little data in public leaderboard.

    * Overfitted

    * Choose wrong splitting strategy

    * Train and test data are from different distributions.

      Sol: calculate mean from the train data, calculate mean for test by "leaderboard probing"

* LB shuffle due to:

  * Randomness

    <u>e.g. "Two Sigma Financial Model and Challenge competition", "Liberty Mutual"</u>

  * Little amount of data

    <u>e.g. "Restaurant Revenue Prediction Competition"</u>

  * Different public/ private distributions

    <u>e.g. "Rossmann Stores Sales competition" (time series)</u>
