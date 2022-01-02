# Tricks and tips

* Data loading
  * Do basic preprocessing and convert csv/txt files into hdf5/npy for much faster loading.
    * Labeling, encoding, category recovery, joining additional data -> hdf5 for pandas data frames or npy for numpy array.
  * Do not forget by default data is stored in 64 bit arrays, most of the times you can safely downcast it to 32 bits.
    * For 2-fold memory saving
  * Large datasets can be processed in chunks.

* Feature engineering

  * See kaggler: cpmp, raddar, giba dmitry larko. How they deal with tabular data (with excel).
  * `rapids.ai` for tabular data

* Performance validation

  * Use train-test split rather than full cross validation loop
  * Use full CV when it is really needed

* Baseline model

  * Start with LightGBM, find some good parameters, evaluate performance of features.
  * Use early stopping, so do not need to tune number of boosting iterations.
  * Do not waste time to fit complex models like neural networks, instead, go tune the model and stacking only when the feature engineering is done. 
  * Focus on the data, feature, domain knowledge, rather than the code. Do not waste time on creating classes, functions, personal frame box, etc. Keep things simple and only save important things.

* Initial pipeline

  * Start from random forest rather than XGBoost, because RF is faster and requires almost no tuning.
  * Create pipeline from simple to complex, from reading data to writing submission file.
  * Fix all random seed, write down exactly how many features were generated, use readable variable names.

* General plan

  * Start with EDA and baseline model
  * Add features in bulks
    * Create all features, as many as possible
    * Evaluate many features all together, rather than one by one.
  * Hyperparameters optimization
    * Find parameters overfit train dataset, then change the parameter to constrain the model. (e.g. large n_estimator would make XGBoost overfit the data)

* Code organization

  * Have reproducible results, keep important code clean.

  * Feature need to be prepared and transformed by the same code in order to guarantee that they are produced in a consistent manner. So move code to separate functions.

  * Use version control system - git

  * Sometime need to restart the notebooks to avoid mistakes. Always restart the kernel and run all before creating a submission.

  * Create one separate notebook for every submission, so we can run the previous solution and compare.

  * A convenience way to validation models: 

    * Split train.csv into train and val with structure of train.csv adn test.csv

      ```python
      train = pd.read_csv("data/train.csv")
      test = pd.read_csv("data/test.csv")
      
      from sklearn.model_selection import train_test_split
      
      train_train, train_val = train_test_split(train, random_state = 660)
      
      train_train.to_csv("data/val/train.csv")
      train_val.to_csv("data/val/val.csv")
      ```

    * When validating, run the code at the top of the notebook

      ```python
      train_path = "data/val/train.csv"
      test_path = "data/val/val.csv"
      ```

    * To retrain models on the whole dataset and get predictions for test set just change

      ```python
      train_path = "data/train.csv"
      test_path = "data/test.csv"
      ```

  * Use macros for a frequent code, load everything

    ```python
    # save code
    %macro -q __imp 1
    
    # store code
    %store __imp
    
    # list all stored macros
    %store
    
    # call the macros
    __imp
    ```

  * Develop custom library with frequent operations implemented

    * Out of fold predictions
    * Averaging 
    * Specify a classifier by it's name

## Example of pipeline

1. Understand the problem

   * Type of problem
   * How big is the data
   * Hardware needed (CPU, GPU, etc)
   * Software needed (PyTorch, sklearn, LightGBM, XGBoost)
   * What is the metric being tested on?

2. EDA

   * Plot histograms of variables. Check that a feature looks similar between train and test.
   * Plot features vs target variable and vs time if available.
   * Consider univariate predictability metrics (IV, R, AUC)
   * Binning numerical features and correlation matrices.

3. Define cv strategy

   * Create a validation approach that best resembles what we are being tested on.
     * Check whether time is important. Split by time, do time based validation. 
     * Check whether there is different entities in the test data than the train data. Do stratified validation
     * Random validation (random k-fold CV)

4. Feature engineering

   The type of problem defines the feature engineering

   * <u>Image classification</u>: scaling, shifting, rotations, CNNs. E.g. previous data science bowls
   * <u>Sound classification</u>: Fourier, Mfcc, specgrams, scaling. E.g. TF speach recognition.
   * <u>Text classification</u>: Tf-idf, svd, stemming, spelling checking, stop words' removal, x-grams. E.g. StumbleUpon Evergreen Classification.
   * <u>Time series</u>: Lags, weighted averaging, exponential smoothing. E.g. Walmart recruitment.
   * <u>Categorical</u>: target encoding, frequency, one-hot, ordinal, label encoding. E.g. Amazon employee.
   * <u>Numerical</u>: scaling, binning, derivatives, outlier removals, dimensionality reduction. E.g. Africa soil.
   * <u>Interactions</u>: multiplications, divisions, group-by features. Concatenations. E.g. Homesite.
   * <u>Recommenders</u>: features on transactional history. Item popularity, frequency of purchase. E.g. Acquire Valued Shoppers.

   Trick is to go back to similar competitions and see what competitors have done.

5. Modeling

   * <u>Image classification</u>: CNNs (ResNet, VGG, densenet)
   * <u>Sound classification</u>: CNNs (CRNN), LSTM
   * <u>Text classification</u>: GBMs, Linear, DL, Naive bayes, KNNs, LibFM, LIBFFM
   * <u>Time series</u>: Autoregressive models, ARIMA, linear, GBMs, DL, LSTMs
   * <u>Categorical</u>: GBMs, Linear models, DL, LiBFM, libFFm
   * <u>Numerical</u>: GBMs, Linear models, DL, SVMs
   * <u>Interactions</u>: GBMs, Linear models, DL
   * <u>Recommenders</u>: CF, DL, LibFM, LIBFFM, GBMs.

6. Ensembling

   * Small data requires simpler ensemble techniques like averaging
   * We can apply varies ensemble techniques to large data 

7. Submissions

   * Check the correlation between different submissions

