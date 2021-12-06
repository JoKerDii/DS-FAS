[TOC]

# Lecture 9

* **Multinomial LRT**

  > **Let $\Theta = (p_1, p_2, p_3, p_4, p_5)$, $\sum_{i=1}^5 p_j= 1$, $P_\theta(x_i = j) = p_j$. Consider testing $H_0: p_1 = p_2 = p_3$ and $p_4 = p_5$ vs. $H_1: H_0$ is not true.** 
  >
  > (Step 1) 
  >
  > Under $H_0$, $h(p) = \begin{pmatrix}h_1(p) \\ h_2(p) \\ h_3(p)\end{pmatrix} = \begin{pmatrix}p_1- p_2 \\ p_1 - p_3 \\ p_4 - p_5\end{pmatrix} = 0$
  >
  > The reduced number of degrees of freedom or number of free parameters is $ r=3$.
  >
  > (Step 2)
  >
  > PMF: $p(X=x) = p^x(1-p)^{1-x} = p_1^{I(x_i=1)}...p_5^{I(x_i=5)}$
  >
  > The likelihood is
  > $$
  > L(p) = \prod_{i=1}^n p_1^{I(x_i=1)}...p_5^{I(x_i=5)} = p_1^{\sum I(x_i=1)}...p_5^{\sum I(x_i=5)}
  > $$
  > Let $q_j$ be the number of $j$s getting from $x_1, ..., x_n$, then $q_j =\sum^n_{i=1} I(x_i = j), j = 1, ..., 5$.
  >
  > The log likelihood is
  > $$
  > \ell(p) = q_1 \log p_1 + ... + q_5 \log p_5
  > $$
  > (Step 3)
  >
  > Take the derivative and solve for regular MLEs for $p_1, ..., p_5$.
  > $$
  > {d\ell(p)\over dp_1} = {q_1\over p_1} - {q_5\over 1- \sum^4_{i=1}p_i} = 0, ...,{d\ell(p)\over dp_4} = {q_4\over p_4} - {q_5\over 1- \sum^4_{i=1}p_i} = 0\\
  > {q_1\over p_1} = {q_2\over p_2} = {q_3\over p_3} = {q_4\over p_4} = {q_5\over 1-\sum^4_{i=1} p_i} = {q_1 + ... + q_5\over p_1+ ... + p_5} = {n \over 1} = n
  > $$
  > We get MLEs
  > $$
  > \widehat{p_1} = {q_1 \over n},\ ...,\ \widehat{p_4} = {q_4 \over n},\ \widehat{p_5} = 1- \sum^4_{i=1}\widehat{p_i}
  > $$
  > (Step 4)
  >
  > Under $H_0$, we have $\begin{cases} p_1 = p_2 = p_3 \\ p_4 = p_5 \\ \sum^5_{i=1} p_i = 1 \end{cases}$, then we get $p_4 = {1 - 3 p_1\over 2}$.
  >
  > The log likelihood under the $H_0$ can be written as
  > $$
  > \ell(p) = q_1 \log p_1 + q_2 \log p_2 + ... + q_5 \log p_5 = (q_1 + q_2 + q_3) \log p_1 + (q_4 + q_5) \log ({1-3p_1\over 2})
  > $$
  > Solve for the $\check{p_i}$,
  > $$
  > {d\ell(p) \over d p_1} = {q_1 + q_2 + q_3 \over p_1} - {3(q_4 + q_5)\over 1-3p_1} = 0\\
  > \implies \check{p_1} = {q_1 + q_2 + q_3 \over 3n} = \check{P_2} = \check{P_3},\  \check{P_4} = {1 - 3 \check{P_1} \over 2} = {q_4 + q_5\over 2n}
  > $$
  > (Step 5)
  >
  > The likelihood ratio is
  > $$
  > \begin{aligned}
  > -2\log\lambda(x) &= -2(\ell(\check{p}) - \ell(\widehat{p})) = 2(\ell(\widehat{p}) - \ell(\check{p}))\\
  > &= 2(q_1 \log ({3q_1\over q_1 + q_2 + q_3}) + q_2 \log ({3q_2\over q_1 + q_2 + q_3}) + q_3 \log ({3q_3\over q_1 + q_2 + q_3})\\ 
  > & + q_4 \log ({2q_4\over q_4 + q_5}) + q_5 \log ({2q_5\over q_4 + q_5}))
  > \end{aligned}
  > $$

# Lecture 10

* **Multivariate Wald Test**

  Testing $H_0: h(\theta)=0, H_1 : h(\theta) \neq 0$. Denote $H(\Theta) = {\partial h(\theta) \over \partial (\theta)}$ in dimension $(r \times p)$. $r:$ number of reduced free parameters or degrees of freedom. $p:$ number of parameters in the parameter space.

  Under $H_0$ and regularity conditions for MLE normality, we have 
  $$
  \sqrt{n} h(\widehat{\theta})\xrightarrow[]{d} \mathcal{N}(0, H(\theta) I(\theta)^{-1} H(\theta)^T)
  $$
  Th us we can construct the Wald test statistic
  $$
  T = n h (\widehat{\theta})^T \{ H(\widehat{\theta}) I(\widehat{\theta})^{-1} H(\widehat{\theta})^T\}^{-1}h(\widehat{\theta})
  $$
  Which converges to $\chi_r^2$ when $n \rightarrow \infty$.

  > **Example:**
  >
  > **Let $\Theta = (p_1, p_2, p_3, p_4, p_5)$, $\sum_{i=1}^5 p_j= 1$, $P_\theta(x_i = j) = p_j$. Consider testing $H_0: p_1 = p_2 = p_3$ and $p_4 = p_5$ vs. $H_1: H_0$ is not true.** 
  >
  > (Step 1)
  >
  > Under $H_0$, $h(p) = \begin{pmatrix}h_1(p) \\ h_2(p) \\ h_3(p)\end{pmatrix} = \begin{pmatrix}p_1- p_2 \\ p_1 - p_3 \\ p_4 - p_5\end{pmatrix} = 0$
  >
  > (Step 2)
  >
  > $H(p) = {\partial h(p) \over \partial p} = \begin{bmatrix} {\partial h_1(p)\over \partial{p_1}} & ... & {\partial h_1(p)\over \partial{p_5}} \\ \vdots & & \vdots \\{\partial h_3(p)\over \partial{p_1}} & ... & {\partial h_3(p)\over \partial{p_5}} \end{bmatrix} =  \begin{bmatrix}1 & -1 & 0 & 0 \\ 1 & 0 & -1 & 0 \\ 1 & 1 & 1 & 2 \end{bmatrix}$
  >
  > (Step 3)
  >
  > The regular MLE for $p_i$,
  > $$
  > \widehat{p_1} = {q_1 \over n},\ ...,\ \widehat{p_4} = {q_4 \over n},\ \widehat{p_5} = {q_5 \over n}
  > $$
  > (Step 4)
  >
  > The Fisher information
  >
  > $I(p) = -E[{\partial ^2 \log f(x|p)\over \partial ^2 p }]$ which is a $4 \times 4$ matrix.
  >
  > (Step 5)
  >
  > The test statistics
  > $$
  > T = h(\widehat{p})^T \{H(\widehat{p} )I(\widehat{p})^{-1} H(\widehat{p})^{T}\}^{-1} h(\widehat{p})\\
  > \text{Dimensions: }(1\times 3)(3\times 4)(4\times 4)(4\times 3)(3\times 1)~~~~~~~~~~~~~~~
  > $$
  > Note that the dimensions should be 
  > $$
  > ~~~~~~(1\times r)(r\times p)(p\times p)(p\times r)(r\times 1)
  > $$

  > **Lecture case 1**
  >
  > (Step 1)
  > $$
  > h(p) = p_1 - p_2
  > $$
  > (Step 2)
  > $$
  > H(p) = \left({d h\over dp_1} \quad {d h\over dp_2}\right) = \left(1 \quad -1\right)
  > $$
  > (Step 3)
  >
  > Log likelihood
  > $$
  > \ell(p) = \sum^{n_1}_{i=1} x_i \log p_1 + \sum^{n_1}_{i=1} (1-x_i) \log (1-p_1) + \sum^{n_2}_{j=1} y_i \log p_2 + \sum^{n_2}_{j=1} (1-y_j)\log(1-p_2)
  > $$
  > Solve for MLEs
  > $$
  > {\partial \ell(p) \over \partial p_1} = {\sum x_i\over p_1} - {\sum (1-x_i)\over 1-p_1} = 0\\
  > {\partial \ell(p) \over \partial p_2} = {\sum y_i\over p_2} - {\sum (1-y_i)\over 1-p_2} = 0\\
  > \implies \widehat{p_1} = {\sum x_i \over n_1}= { x_{11} \over n_1},\ \widehat{p_2} = {\sum y_i \over n_2}= { x_{21} \over n_2}
  > $$
  > (Step 4)
  >
  > Find Fisher Information,
  > $$
  > \begin{aligned}
  > {\partial^2 \ell(p) \over \partial^2 p_1}& = -{\sum x_i\over p_1^2} - {\sum (1-x_i)\over (1-p_1)^2}\\
  > {\partial^2 \ell(p) \over \partial^2 p_2}& = -{\sum y_i\over p_2^2} - {\sum (1-y_i)\over( 1-p_2)^2}\\
  > {\partial^2 \ell(p) \over \partial p_1 \partial p_2}& = 0\\
  > 
  > -E[{\partial^2 \ell(p) \over \partial^2 p_1} ]& = {n_1 \over p_1} + {n_1 \over 1-p_1}\\
  > -E[{\partial^2 \ell(p) \over \partial^2 p_2} ]& = {n_2 \over p_2} + {n_2 \over 1-p_2}\\
  > -E[{\partial^2 \ell(p) \over \partial p_1 \partial p_2}] & = 0
  > \end{aligned}
  > $$
  > So the Fisher information is
  > $$
  > I(p) = \begin{pmatrix} {n_1 \over p_1(1-p_1)} & 0 \\ 0 & {n_2 \over p_2(1-p_2)} \end{pmatrix}
  > $$
  > The inverse of Fisher information is
  > $$
  > I(p)^{-1} = \begin{pmatrix} {p_1(1-p_1) \over n_1} & 0 \\ 0 & { p_2(1-p_2)\over n_2} \end{pmatrix}
  > $$
  > (Step 5)
  > $$
  > H(p) I(p)^{-1} H(p)^T = \begin{pmatrix} 1 & -1 \end{pmatrix}\begin{pmatrix} {p_1(1-p_1) \over n_1} & 0 \\ 0 & { p_2(1-p_2)\over n_2} \end{pmatrix} \begin{pmatrix} 1 \\ -1 \end{pmatrix} = {p_1(1-p_1) \over n_1} + { p_2(1-p_2)\over n_2}
  > $$
  > The Wald Test statistic is
  > $$
  > \begin{aligned}
  > T &= {h(\widehat{p}) \over \sqrt{H(\widehat{p}) I(\widehat{p})^{-1} H(\widehat{p})^T}}\\
  > &= {\widehat{p_1} - \widehat{p_2} \over \sqrt{{\widehat{p_1}(1-\widehat{p_1}) \over n_1} + { \widehat{p_2}(1-\widehat{p_2})\over n_2}}}
  > \end{aligned}
  > $$
  > where $\widehat{p_1} = {\sum x_i \over n_1}= { x_{11} \over n_1},\ \widehat{p_2} = {\sum y_i \over n_2}= { x_{21} \over n_2}$.

  > **Lecture case 2**
  >
  > (Step 1)
  > $$
  > h(p) = \log {p_1(1-p_2) \over p_2(1-p_1)} = \log p_1 + \log (1-p_2) - \log p_2 - \log (1-p_1)
  > $$
  > (Step 2)
  > $$
  > H(p) = \left({d h\over dp_1} \quad {d h\over dp_2}\right) = \left({1\over p_1 (1-p_1)} \quad {1\over p_2 (1-p_2)}\right)
  > $$
  > (Step 3)
  >
  > Take the derivative of the log likelihood and get MLE
  > $$
  > \widehat{p_1} = {\sum x_i \over n_1}= { x_{11} \over n_1},\ \widehat{p_2} = {\sum y_i \over n_2}= { x_{21} \over n_2}
  > $$
  > (Step 4)
  >
  > Get Fisher Information from log likelihood
  > $$
  > I(p)^{-1} = \begin{pmatrix} {p_1(1-p_1) \over n_1} & 0 \\ 0 & { p_2(1-p_2)\over n_2} \end{pmatrix}
  > $$
  > (Step 5)
  > $$
  > H(p) I(p)^{-1} H(p)^T = \begin{pmatrix} {1\over p_1 (1-p_1)} & {1\over p_2 (1-p_2)} \end{pmatrix}\begin{pmatrix} {p_1(1-p_1) \over n_1} & 0 \\ 0 & { p_2(1-p_2)\over n_2} \end{pmatrix} \begin{pmatrix} {1\over p_1 (1-p_1)} \\ {1\over p_2 (1-p_2)} \end{pmatrix} = {1\over n_1p_1 (1-p_1)} + {1\over n_2 p_2 (1-p_2)}
  > $$
  > The Wald Test statistic is
  > $$
  > \begin{aligned}
  > T &= {h(\widehat{p}) \over \sqrt{H(\widehat{p}) I(\widehat{p})^{-1} H(\widehat{p})^T}}\\
  > &= {\log {\widehat{p_1} (1-\widehat{p_2}) \over \widehat{p_2} (1-\widehat{p_1}) } \over \sqrt{{1 \over n_1 \widehat{p_1}} + {1 \over n_1 (1-\widehat{p_1})} + {1 \over n_2 \widehat{p_2}} + {1 \over n_2 (1-\widehat{p_2})} } }\\
  > &= {\log {\widehat{p_1} (1-\widehat{p_2}) \over \widehat{p_2} (1-\widehat{p_1}) } \over \sqrt{{1 \over x_{11}} + {1 \over x_{12}} + {1 \over x_{21}} + {1 \over x_{22}} } }
  > \end{aligned}
  > $$

  

  
