# Evaluation Metrics

## Evaluation Metrics

If you model is scored with some metric, you get best results by optimizing exactly that metric.

* Regression

  * Mean square error (MSE)
    $$
    MSE = {1\over N}\sum^N_{i=1}(y_i - \overline{y})^2
    $$

  * Root mean square error (RMSE)

    * Since square root is non-decreasing, if our target metric is RMSE, we can still compare our models using MSE, since MSE will order the models in the same way as RMSE. We can optimize MSE instead of RMSE.

    * MSE is easier to work with. 

    * There is one difference between the two for gradient-based models.
      $$
      {\partial RMSE \over \partial \widehat{y_i}} = {1\over 2\sqrt{MSE}} {\partial MSE \over \partial \widehat{y_i}}
      $$
      This means traveling along MSE gradient is equivalent to traveling along RMSE gradient but with a different learning rate and the learning rate depends on MSE value itself.

      So although RMSE and MSE are similar in terms of models scoring, they cannot be immediately interchangeable for gradient based methods. We will need to adjust some parameters like the learning rate.

  * R-squared
    $$
    R^2 = 1 - {MSE \over {1\over N} \sum^N_{i=1}(y_i - \overline{y})^2}
    $$
    When $MSE = 0$, $R^2 = 1$, when $MSE = $ MSE over a constant model, $R^2 = 0$.

    We can optimize $R^2$ instead of optimizing $MSE$.

  * Mean absolution error (MAE)

    More robust than MSE. Not sensitive to errors. 

    Often used in finance.

    The gradient of MAE is a step function and it takes $-1$ when the $\widehat{y} < y$ and $+1$ when the $\widehat{y} > y$.

  * MSPE (weighted version of MSE)
    $$
    MSPE = {100\% \over N} \sum^N_{i=1}\left({y_i - \widehat{y_i}\over y_i}\right)^2
    $$

  * MAPE (weighted version of MAE)
    $$
    MAPE = {100\% \over N} \sum^N_{i=1}\left|{y_i - \widehat{y_i}\over y_i}\right|
    $$

  * (R)MSLE: root mean square logarithmic error (MSE in log space)
    $$
    \begin{aligned}
    RMSLE &=  \sqrt{{1\over N} \sum^N_{i=1} (\log (y_i + 1) - \log(\widehat{y_i} + 1))^2}\\
    &= RMSE(\log(y_i+1), \log(\widehat{y_i} + 1))\\
    &= \sqrt{MSE(\log(y_i+1), \log(\widehat{y_i}+1))}
    \end{aligned}
    $$

* Classification

  * Some notations:

    Soft labels (soft predictions) are classifier's scores

    Hard labels (hard predictions) can be $argmax_i f_i(x)$ or $[f(x) > b]$, b - threshold.

  * Accuracy score
    $$
    Acc = {1\over N} \sum^N_{i=1}[\widehat{y} = y_i]
    $$

  * Logarithmic loss (logloss)

    * Binary
      $$
      LogLoss = - {1\over N} \sum^N_{i=1} y_i \log (\widehat{y_i}) + (1-y_i) \log(1- \widehat{y_i})
      $$

    * Multiclass
      $$
      LogLoss = -{1\over N} \sum^N_{i=1}\sum^L_{i=1} y_{il} \log (\widehat{y_{il}})
      $$

    * In practice
      $$
      LogLoss = -{1\over N} \sum^N_{i=1} \sum^L_{i=1} y_{il} \log(\min(\max(\widehat{y_{il}}, 10^{-15}), 1-10^{-15}))
      $$

    Logloss strongly penalizes completely wrong answers.

  * Area Under Curve(AUC ROC)

    * Only for binary tasks
    * Depends only on ordering of the predictions, not on absolute values
    * Several explanations
      * Area under curve
      * Pairs ordering

  * (Quadratic weighted) Kappa

## Metrics optimization

Difference between metrics and loss:

* Target metric is what we want to optimize
* Optimization loss is what model optimizes

Approaches for target metric optimization

* MSE, Logloss

* MSPE, MAPE, RMSLE: 

  MSPE metric cannot be optimized directly with XGBoost, instead we can optimize MSE loss instead which XGBoost can optimize.

* Accuracy, Kappa

* Custom loss function (especially for XGBoost)

  We can define an objective, function that computes first and second order derivatives with respect to predictions.

Early stopping

* Optimize metric M1, monitor metric M2, stop when M2 score is the best.



