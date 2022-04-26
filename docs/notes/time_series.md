# I. Introduction

### About the promise of neural networks and deep learning methods in general for time series forecasting.

<u>Limitations of classical methods:</u>

1. Focus on complete data, not missing data.
2. Strong assumption of linear relationship.
3. Focus on fixed temporal dependence, while the relationship between observations at different times, and in turn the number of lag observations provided as input, must be diagnosed and specified.
4. Focus on univariate data, does not work for multiple input variables.
5. Focus on one-step forecasts, does not forecast with a long time horizon.

<u>Merits of simple NN (e.g. MLP):</u>

1. Robust to noise in input data and able to support learning and prediction in the presence of missing values.
2. Able to learn linear and nonlinear relationships, no strong assumption about mapping function. 
3. Allow multivariate inputs and support multivariate forecasting. 
4. Allow multi-step forecasts by specifying an arbitrary number of output values.

<u>Limitations of FNN:</u>

1. Fixed number of lag input variables.
2. Fixed number of output variables.

<u>Strength of CNN:</u>

*Representation learning* is when model learns how to automatically extract the features from the raw data that are directly useful for the problem. CNN achieves this by extracting features of how they occur in the data, so-called *transform* or *distortion invariance*. CNNs combine three architectural ideas to ensure some degree of shift and distortion invariance: local receptive fields, shared weights (or weigh replication), and spatial or temporal subsampling.

<u>Strength of CNN on time series:</u>

A sequence of observations can be treated as a 1d image read into CNN. A variety of processing units can yield an effective representation of local salience of the signals. The deep architecture allows multiple layers of these processing units to be stacked, so that it can characterize the salience of signals in different scales.

1. Feature extraction is performed in task dependent and non hand-crafted manners. The features extracted have more discriminative power with respect to the classes of human activities. Feature extraction and classification are unified in one model so their performances are mutually enhanced
2. Suport multivariate input, multivariate output
3. Can learn arbitrary but complex functional relationships.

<u>Limitation of CNN on time series:</u>

1. Cannot learn directly from lag observations.
2. Cannot handle the order between observations when learning a mapping function from inputs to outputs.
3. Use fixed size time windows.

<u>Strength of LSTM (can be used in time series):</u>

1. Capable of learning the complex inter-relationships between words both within a given language and across languages in translating form on language to another.
2. Automatically learn the *temporal dependence* from the data. *Learned temporal dependence*: the most relevant context of input observations to the expected output is learned and can change dynamically.
3. Able to learn long-term correlations in a sequence. Does not require pre-specified time window.

<u>Strength of NN:</u>

Deep learning methods, such as Multilayer Perceptrons, Convolutional Neural Networks, and Long Short-Term Memory Networks, can be used to automatically learn the temporal dependence structures for challenging time series forecasting problems. Neural networks may be the best solution for the time series forecasting problems when classical methods fail and when machine learning methods require elaborate feature engineering, deep learning methods can be used with great success.

Traditionally, MLPs have been used for time series forecasting. Currently, most promising methods in DL in time series forecasting are CNNs, LSTMs, and hybrid models.

1. NN learns arbitrary mapping functions
2. NN may not require a scaled or stationary time series as input.
3. NN supports multivariate inputs
4. NN supports multi-step outputs
5. CNNs are more efficient in feature learning than MLP
6. RNNs are more efficient in automatically learning the temporal dependencies both within the input sequence and from the input sequence to the output.
7. LSTM is more efficient in learning of temporal dependencies than other NNs.
8. Hybrid models efficiently combine the diverse capabilities of different architectures. (e.g. CNN-LSTMs)

### A taxonomy and framework of questions for systematically identifying the properties of a time series forecasting problem.

1. Inputs and outputs

2. Endogenous or exogenous

   *Endogenous*: Input variables that are influenced by other variables in the system and on which the output variable depends.

   *Exogenous*: Input variables that are not influenced by other variables in the system and on which the output variable depends.

3. Unstructured or structured

   *Unstructured*: No obvious systematic time-dependent pattern in a time series variable.

   *Structured*: Systematic time-dependent patterns in a time series variable (e.g. trend and/or seasonality).

4. Regression or classification

5. Univariate or multivariate

   *Univariate*: One variable measured over time.

   *Multivariate*: Multiple variables measured over time.

   *Univariate and Multivariate Inputs*: One or multiple input variables measured over time.

   *Univariate and Multivariate Outputs*: One or multiple output variables to be predicted.

6. Single-step or multi-step

   *One-step*: Forecast the next time step.

   *Multi-step*: Forecast more than one future time steps.

7. Static or dynamic

   *Static*. A forecast model is fit once and used to make predictions.

   *Dynamic*. A forecast model is fit on newly available data prior to each prediction.

8. Continuous or discontinuous

   *Contiguous*. Observations are made uniform over time (e.g. one observation each hour, day, month or year.).

   *Discontiguous*. Observations are not uniform over time. The lack of uniformity of the observations may be due to missing or corrupt values. In this case, specific data formatting might be required when fitting some models to make the observations uniform overtime.

### A systematic four-step process to work through a new time series forecasting problem to get the most out of naive, classical, machine learning and deep learning forecasting methods (model evaluation, model selection, etc.)

1. Define problem

   * Identify properties by using the framework above
   * Data visualizations (e.g. ACF and PACF for seasonal time)
   * Statistical analysis
   * Domain knowledge

2. Design test harness

   * Split the dataset into a train and test set
   * Fit a candidate approach on the train set
   * Make predictions on the test set or using walk-forward validation
   * Calculate a metric that measures the prediction performance.

3. Test models

   A list of models based on univariate time series forecasting problem. (VAR/VARMA for multivariate time series forecasting)

   * Baseline
   * Autoregression (e.g. Box-Jenkins process, SARIMA method)
   * Exponential smoothing (e.g. single, double, triple)
   * Linear machine learning (e.g. linear regression with regularization)
   * Nonlinear machine learning (e.g. kNN, decision trees, SVR)
   * Ensemble machine learning (e.g. random forest, gradient boosting, stacking)
   * Deep learning (MLPs, CNNs, LSTMs, and hybrid model)

   Given more time, we can:

   * Search model configurations
   * Search model hyperparameters
   * Better data preparation
     * Differencing to remove a trend 
     * Seasonal differencing to remove seasonality 
     * Standardize to center 
     * Normalize the rescale 
     * Power transform to make normal
   * Better feature engineering
   * Explore more complex models
   * Ensembles of base models

   To speed up the evaluation process:

   * Parallel computing via cloud hardward (Amazon EC2)
   * Reduce

4. Finalize models

### Transform time series data into samples with input and output in order to train a supervised learning algorithm

<u>Supervised machine learning</u>

* Sliding window method / lag method

  The use of prior time steps to predict the next time step. The number of previous time steps is called the window width or size of the lag. 

  Sliding window is the basis to turn any time series dataset into a supervised learning problem. It can be used for multivariate time series data, and multi-step forecast.

<u>Deep Learning (Chapter 6, 7, 8, 9)</u>

The 2D structure of the supervised learning data must be transformed to a 3D structure for CNN and LSTM, by splitting a long time series into subsequences. The expected 3D structure of input data is [samples, timesteps, features]. 

* Samples. One sequence is one sample. A batch is comprised of one or more samples.
* Time Steps. One time step is one point of observation in the sample. One sample is comprised of multiple time steps.
* Features. One feature is one observation at a time step. One time step is comprised of one or more features.

`model.add(LSTM(32, input_shape=(3, 1)))` 

* 32: number of units in the first hidden layer
* 3: number of time steps (LSTM works better with 200-400 time steps)
* 1: number of features

Example: Suppose there is a dataset, with 5000 time steps and one feature. The shape of the 3D data would be [25, 200, 1], after splitting 5000 time steps into 25 shorter sub-sequence, each with 200 time steps. 

### Naive and the top performing classical methods such as SARIMA and ETS

<u>Simple strategies (Chapter 11, 17)</u>

* Naive
* Average

<u>Autoregressive models</u>

* *Autoregressive Integrated Moving Average (ARIMA)*

  Can handle data with a tread, but does not support time series with seasonal component.

* *Seasonal Autoregressive Integrated Moving Average (SARIMA)* (Chapter 13, Chapter 18)

  Method for time series forecasting with univariate data containing trends and seasonality. It adds (based on ARIMA) three new hyperparameters to specify the autoregression (AR), differencing (I) and moving average (MA) for the seasonal component of the series, as well as an additional parameter for the period of the seasonality.

  * AR: *Autoregression*. A model that uses the dependent relationship between an observation and some number of lagged observations.
  * I: *Integrated*. The use of di erencing of raw observations (e.g. subtracting an observation from an observation at the previous time step) in order to make the time series stationary.
  * MA: *Moving Average*. A model that uses the dependency between an observation and a residual error from a moving average model applied to lagged observations.

<u>Exponential smoothing methods (Chapter 12)</u>

* Used for univariate data that can be extended to support data with a systematic trend or seasonal component.
* Similar to ARIMA in that a prediction is a weighted sum of past observations, but the model uses an exponentially decreasing weight for past observations. Specifically, past observations are weights with a geometrically decreasing ratio, i.e. more recent observations have higher weights. 
* Sometimes referred to as *Error Trend and Seasonality (ETS)* models. 
* Three types of exponential smoothing
  * *Simple Exponential Smoothing (SES)*: for univariate data without a trend or seasonality.
  * *Double Exponential Smoothing (DES)*: Explicitly handle trend in univariate time series. 
  * *Triple Exponential Smoothing (TES)*: Advanced that support seasonality

### Develop MLP, CNN, LSTM, and hybrid NN models for time series forecasting

<u>MLP</u>

* Univariate MLP

  The expected shape of input data is [sample, feature]. The input dimension of MLP is the number of features, or the number of time steps per sample. The output dimension is 1.

* Multivariate MLP

  * Multiple input series

    Split and reshape data into [sample, time step, feature]. For MLP, flatten the data into [sample, time step * feature].

  * Multi-headed MLP

    Each input series can be handled by a separate MLP and the output of each of these submodels can be combined before a prediction is made for the output sequence.

    e.g. if there are 2 features [sample, time steps, 2], split the 3D input data into two separate arrays: [sample, time step].

  * Multiple parallel series

    Multiple parallel series is the case where a value must be predicted for each of the time series. This refers to multivariate forecasting.

    The data is of the shape [sample, time step, feature] and reshaped before input to [sample, time step * feature]. The output is the one time step of all variables, which is of the shape [1, feature]. 

  * Multi-output MLP

    Each output series can be handled by a separate output MLP model. 

    e.g. we can define one output layer for each of these three series, where each output submodel will forecast a single time step. So when training the model, we'd better convert the output from shape [sample, 3] to three arrays of [sample, 1]. 

* Univariate Multi-step MLP

  Prepare training data: split a given univariate time series into samples with a specified number of input and output time steps. 

* Multivariate Multi-step MLP

  * Multiple input multi-step output
  * Multiple parallel input and multi-step output

<u>How to forecast univariate, multivariate, multi-step, and multivariate multi-step time series forecasting problems in general.</u>

<u>How to transform sequence data into a three-dimensional structure in order to train convolutional and LSTM neural network models.</u>

<u>How to grid search deep learning model hyperparameters to ensure that you are getting good performance from a given model.</u>

<u>How to prepare data and develop deep learning models for forecasting a range of univariate time series problems with di erent temporal structures.</u>

<u>How to prepare data and develop deep learning models for multi-step forecasting a real-world household electricity consumption dataset.</u>

<u>How to prepare data and develop deep learning models for a real-world human activity recognition project.</u>

