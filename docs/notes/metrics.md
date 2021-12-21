# Evaluation Metrics and Mean Encoding

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

    

