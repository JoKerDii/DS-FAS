# Machine Learning Concept Q&A (III)

## 1. ANN

* **Types of output units? **

  * Output unit for bernoulli - <u>sigmoid</u>:
    $$
    \widehat{Y} = {1\over 1 + e^{\phi(X)}}
    $$

  * Output unit for multi-noulli - <u>softmax</u>:
    $$
    \widehat{Y} = {e^{\phi_k(X)} \over \sum^K_{i=1} e^{\phi_k(X)}}
    $$

* **Types of loss function?**

  To find the loss function for a given distribution $p(y_i|W;x_i)$, first compute the likelihood for all points
  $$
  L(W;X;Y) = p(Y|W;X) = \prod_i p(y_i|W;x_i)
  $$
  and then maximize the likelihood is equivalent to minimizing the negative log-likelihood.
  $$
  \ell(W;X,Y) = -\log L(W;X,Y) = - \sum_i \log p (y_i | W;x_i)
  $$

  * Distribution is normal then negative log likelihood is <u>MSE</u>
    $$
    \ell(W; X,Y) = \sum_i (y_i - \widehat{y_i})^2
    $$

  * Distribution is bernoulli then negative log likelihood is <u>binary cross-entropy</u>
    $$
    \ell(W; X,Y) = -\sum_i [y_i \log p_i + (1- y_i)\log(1-p_i)]
    $$


| Output type | Output distribution | Output layer | Cost function        |
| ----------- | ------------------- | ------------ | -------------------- |
| Binary      | Bernoulli           | Sigmoid      | Binary cross entropy |
| Discrete    | Multinoulli         | Softmax      | Cross entropy        |
| Continuous  | Gaussian            | Linear       | MSE                  |
| Continuous  | Arbitrary           |              | GANs                 |

* **How does one minimize a loss function?**

  Minima of loss function must occur at points where the gradient is 0. There are three approaches:

  * <u>Brute force</u>: calculate the loss function for every possible $\beta_i$ and select the $\beta_i$ where the loss function is minimum. But very computationally expensive.
  * <u>Exact</u>: solve the equation
  * <u>Greedy algorithm</u>: gradient descent. The gradient at any point is the direction of steepest increase. The negative gradient at any point is the direction of steepest decrease. By following negative gradient, we can eventually find the lowest point.

* **What is Gradient Descent and how it minimizes loss?**

  Gradient descent is an algorithm for optimization of first order to finding a minimum of a function. It is an iterative method. $L$ is decreasing much faster in the direction of the negative derivative. The derivative is taken by following chain rule. The learning rate is controlled by the magnitude of $\eta$.
  $$
  w^{(i+1)} = w^{(i)} - \eta {d \ell\over dw}
  $$

* **How to choose learning rate and why it matters?**

  The choice of learning rate has a significant impact on the performance of gradient descent. When it is too small, the algorithm makes very little progress. When it is too large, the algorithm may overshoot the minimum and has crazy oscillations. When it is appropriate, the algorithm will find the minimum and the algorithm converges.

  Alternative methods to address how to set or adjust teh learning rate such as using the derivative or second derivatives and the momentum.

* **How can we tell when gradient descent is converging?**

  We can plot the loss function vs. iterations - trace plot. While the loss is decreasing throughout trading, it does not look like descent hit the bottom, this means the learning rate is small. When loss is mostly oscillating between values rather than converging, this is means the learning rate is too large. When the loss has decreased significantly during training, this means the learning rate is appropriate. Towards the end, the loss stabilizes and it cannot decrease further, this means the algorithm converges.

* **Can we guarantee that there is always a global minimum or gradient descent can always find a global minimum?**

  No, we can't. If the function is convex then the stationary point will be a global minimum.

  Linear and polynomial regression loss function are convex (Hessian (2nd derivative) positive semi-definite everywhere). Neural network regression loss functions are not convex. Most stationary points are local minima but not global optima.

* **What would be a good strategy to get the global minimum?**

  We can either randomly restarts or add noise to the loss function.

* **What is stochastic gradient descent?**

  <u>SGD</u> updates weights by subtracting the current weight by a factor (called learning rate) of its gradient. Specially, it considers only one example at a time to take a single step. Within one epoch (epoch: one forward pass and one backward pass of all the training examples), the SGD has steps:

  1. Take one example

  2. Feed it to neural network

  3. Calculate its gradient

  4. Use the gradient to update the weights

     $W_i = W_i - \eta({\partial L_i \over \partial W_i})$

  5. Repeat 1-4 for all examples in training dataset

  The loss function is calculated for each example:
  $$
  L_i = -(y_i \log p_i + (1-y_i)\log(1-p_i))
  $$

* **What are the SGD optimizers?**

  An optimizer is a function or an algorithm that modifies the attributes of the neural network, such as weights and learning rate. Thus, it helps in reducing the overall loss and improve the accuracy.

  There are three variations in the term of SGD: $W_i = W_i - \eta({\partial L_i \over \partial W_i})$

  1. Adapt the gradient components ${\partial L_i \over \partial W_i}$

     e.g. Momentum, Nesterov Momentum

  2. Adapt the learning rate components $\eta$

     e.g. AdaGrad, RMSprop

  3. Both (1) and (2)

     e.g. Adam

* **What is mini-batch stochastic gradient descent?**

  <u>Mini batch SGD</u> uses a subset of examples, rather than a single example. Partition the dataset in mini-batches helps SGD to escape from local minima.
  $$
  L^k = -\sum_{i \in b^k}[y_i \log p_i + (1-y_i)\log(1-p_i)]
  $$

  Within one epoch the mini batch SGD has steps:

  1. Divide data into mini-batches
  2. Pick a mini-batch
  3. Feed it to neural network
  4. Calculate the mean gradient of the mini-batch
  5. Use the mean gradient calculated in step 4 to update the weights
  6. Repeat steps 2-5 for the mini-batches

* **What is back propagation and why?**

  Backpropagation is a process involved in training a neural network. It involves taking the error rate of a forward propagation and feeding this loss backward through the neural network layers to fine-tune the weights. Specifically, it computes the gradient of the loss function for a single weight by the <u>chain rule.</u> It efficiently computes one layer at a time, and each weight can be updated using the gradients. 

* **What are the challenges in optimization in NN?**

  1. <u>No critical points.</u> 

     Ideally we want global minimum, but this might not be possible. Some local minima performs as well as the global one, so it is an acceptable stopping point. Some cost functions do not have critical points.

  2. <u>Exploding gradients.</u> 

     Exploding gradient is a problem when large error gradients accumulate and result in very large updates to the weights in a neural network during training. 

     Signs of exploding gradients:

     * The model weights quickly become very large during training
     * The model weights go to NaN values during training
     * The error gradient values are consistently above 1.0 for each node and layer during training

     One primary cause of gradients exploding lies in too large of a weight initialization and update. Hence, initializing model weights properly is the key to fix this exploding gradients problem. 

     It can also be mitigated using <u>gradient clipping</u>. We set a threshold value, and if the gradient is larger than this threshold, we set it to the threshold.
     $$
     \text{if}\,\, \| {\partial L\over \partial W}\| > u: {\partial L\over \partial W} = \text{sign}({\partial L \over \partial W})u
     $$
     where $u$ is user defined threshold.

  3. <u>Poor conditioning.</u>

     Poorly conditioned Hessian matrix. High curvature that small steps leads to huge increase. Learning is slow despite strong gradients, because of oscillations.

  4. <u>Vanishing gradients</u>

     The vanishing gradient problem happens in neural networks with gradient-based learning methods and backpropagation. In these learning methods, each of the weights of the neural network receives an update proportional to the partial derivative of the loss function with respect to the current weight in each iteration of training. Sometimes when gradients become vanishingly small, this prevents the weight to change value.

     When neural network has many hidden layers, the gradients in the earlier layers will become very low as we multiply the derivatives of each layer. As a result, learning in the earlier layers becomes very slow. This problem of vanishing gradient happens when training neural networks with many layers because the gradient diminishes dramatically as it propagates backward through the network.
     
     Some ways to fix it:
     
     * Use skip / residual connections
     * Use ReLU or Leaky ReLU rather than sigmoid or tanh activation functions
     * Use models that help propagate gradients to earlier time steps like in GRUs and LSTMs

* **What is Momentum?**

  Gradient descent with momentum replaces the current gradient with 'momentum', which is an aggregate of gradients. This aggregate is the <u>exponential moving average of current and past gradients (up to time $t$)</u>.

  Conventional gradient descent depends only on the current gradient to update the weight:
  $$
  g = {1\over m} \sum_i \nabla_WL(f(x_i;W), y_i) \\
  W^* = W - \eta g
  $$
  New gradient descent with momentum: ($\alpha$ controls how quickly effect of past gradients decay)
  $$
  v = \alpha v + (1-\alpha) g\\
  W^* = W - \eta v
  $$
  The <u>intuition</u> behind momentum: take the exponential moving average, where past gradient values are given higher weights (importance) than the current one. Intuitively, discounting the importance given to the current gradient would ensure that the weight update will not be sensitive to the current gradient.

* **What is Nesterov Momentum?**

  Nesterov Momentum has better theoretical converge guarantees converges than conventional momentum. The idea is to look ahead of the weights and apply an interim update: 
  $$
  v = \alpha v + (1-\alpha) g\\
  \tilde{W} = W - \eta v \\
  \tilde{g} = {1\over m} \sum_i \nabla_WL(f(x_i;\tilde{W}), y_i) \\
  \tilde{v} = \alpha \tilde{v} + (1-\alpha) \tilde{g}\\
  W = W - \eta \tilde{v}
  $$

* **What is AdaGrad?** 

  Stands for adaptive gradient, decides the learning rate by the square root of cumulative sum of current and past squared gradients (up to time $t$). While the gradient component remains unchanged compared to the vanilla SGD.

  Use different $\eta_i$ for different $W_i$, that is inversely proportional to the $|g_i|$ for each $W_i$.
  $$
  W_i^* = W_i - \eta_ig_i\\
  \eta_i \propto {1\over |g_i|} = {\epsilon\over \delta + |g_i|}
  $$
  So the new gradient descent with adaptive learning rate is
  $$
  r_i^* = r_i + g_i^2\\
  W_i^* = W_i - {\epsilon \over \delta + \sqrt{r_i}}g_i
  $$
  Where $\delta$ is a smaller floating-point value to ensure that the denominator won't be zero. In kerns, $\delta$ is called the fuzz factor. $r$ initializes to zero.

  The <u>intuition</u> behind adaptive learning rate: If the steps of minimization of loss function is oscilating, the learning will be slower along parameter $W$. With different learning rate for different gradient, we can control the oscillations. We want to get out of the flat area as fast as possible and look for a downward slope that could lead us to a global minimum. Therefore, we want to increase the learning rate component (make it faster) when the magnitude of the gradients is small.

  The drawback of AdaGrad is that for non-convex problems, it can prematurely decrease learning rate, resulting in slow convergence.

* **What is RMSProp and what is the difference from AdaGrad?**

  Stands for root mean square prop. It's very similar to AdaGrad as an adaptive learning rate strategy. The only difference is that RMSProp used <u>exponentially weighted average</u> for gradient accumulation
  $$
  r_i = \rho r_i + (1- \rho)g_i^2\\
  W_i = W_i - {\epsilon \over \delta + \sqrt{r_i}}g_i
  $$

* **What is Adam optimizer?**

  <u>An optimizer is a function or an algorithm that modifies the attributes of the neural network, such as weights and learning rate.</u> 

  Adam is a combination of RMSProp and Momentum. It acts upon

  * The gradient component by using momentum, the exponential moving average of gradients (like in momentum)
  * Th learning rate component by dividing the learning rate by square root of $r$, the exponential moving average of squared gradients (like in RMSprop)

  Estimate first moment $v_i$ and second moment $r_i$, then update the parameters:
  $$
  v_i = \rho_1 v_i + (1-\rho_1)g_i\\
  r_i = \rho_2 r_i + (1- \rho_2)g_i^2\\
  W_i = W_i - {\epsilon \over \delta + \sqrt{r_i}}v_i
  $$

* **Why we need bias correction and how to do that?** *

  1st and 2nd moment gradient estimates are started off with both estimates being zero. Hence those initial values for which the true value is not zero, would bias the result. We are concerned about the bias during this initial phase, while your exponentially weighted moving average is warming up, then bias correction can help you get a better estimate early on.

  We do the following correction before updating weights
  $$
  v_{corr} = {v \over 1 - \rho_1^t}\\
  r_{corr} = {r \over 1 - \rho_2^t}
  $$
  where $t$ is the number of the current iteration.

  For example, Adam uses bias correction to adjust for a slow startup when estimating momentum and a second moment.

* **Why do we care parameter initialization and how to do that?** 

  Weight initialization is used to define the initial values for the parameters in neural network models prior to training the models on a dataset. 

  Initialization can have a significant impact on convergence in training DNNs and performance of DNNs. Its main objective is to prevent layer activation outputs from exploding or vanishing gradients in the training process, so that the network won't take too long to converge. 

  Notice that initializing all weights with zeros would lead the neurons to learn the same features during training. In fact, any constant initialization scheme will perform poorly, because neurons will evolve symmetrically throughout training with identical gradients. 

  Therefore, we have to take care of 

  1) breaking symmetry between units to ensure each unit computes a different function; 
  2) avoiding large values of initialization which leads to <u>exploding gradients</u>, small values of initialization which leads to <u>vanishing gradient</u>; 
     * The mean of the activations should be zero
     * The variance of the activations should stay the same across every layer

  There are several ways of parameter initialization: Xavier initialization, normalized initialization - Kaiming He initialization, etc.

  * <u>Xavier weight initialitzation</u>

    The xavier initialization method calculates weight as a random number with a uniform probability distribution (U) between the range -(1/sqrt(n)) and 1/sqrt(n), where *n* is the number of inputs to the node.

    All weights of layer $\ell$ are initialized randomly from a normal distribution with mean 0 and a same variance across every layer. Biases are initialized with zeros. Xavier initialization works with <u>Tanh or Sigmoid activations</u>.

  * <u>Normalized Xavier weight initialization</u>

    The normalized xavier initialization method calculates weight as a random number with a uniform probability distribution (U) between the range -(sqrt(6)/sqrt(n + m)) and sqrt(6)/sqrt(n + m), where *n* us the number of inputs to the node (e.g. number of nodes in the previous layer) and *m* is the number of outputs from the layer (e.g. number of nodes in the current layer).

  * <u>Kaiming He initialization</u>

    Fill the gap of Xavier's inability for ReLU. He initialization is developed specifically for nodes and layers that use ReLU.

    The he initialization method is calculated as a random number with a Gaussian probability distribution (G) with a mean of 0.0 and a standard  deviation of sqrt(2/n), where *n* is the number of inputs to the node.

  The benefits of these initialization techniques:

  * Avoid exploding or vanishing gradients
  * Avoid slow convergence and ensure not keeping oscillating off the minima.

  Bias initialization includes initializing output unit bias and initializing hidden unit bias. Usually there is no problem to have 0 bias because the gradient wrt bias does not depend on the gradients of the deeper layers.  

* **Why some heuristic weight initializations do not work?**

  * <u>Zero initialization</u> (initialize all weight to 0)

    The derivative wrt loss function is the same for all weights in the same layer. This makes the hidden layers symmetric and this process continues for all n iterations. So the network would perform poorly. Notice that if weights are non-zero, 0 bias won't be problematic since non-zero weights could take care of breaking the symmetry.

  * <u>Random Initialization</u> (initialized weights randomly)

    It can break the symmetry and works better than zero initialization. However, it is easily encounter vanishing gradients or exploding gradients. 

    * Vanishing gradients: weight update is minor which results in slower convergence. In worse case, this would stop neural network from training further. With ReLU, vanishing gradients are not a problem though.
    * Exploding gradients: they would cause a very large change in the value of the overall gradient. This means the change in weights would be huge.

* **What are the common methods of regularization in NN and why?**

  Regularization is any modification to a learning algorithm that is intended to limit the complexity of the algorithm to reduce its generalization error (but not its training error) or avoid overfitting the training data.

  * <u>Norm penalties</u>

    <u>Problems</u>: Large weights in a neural network implies that this is a complex network that has overfit the training data. Large weights make the network unstable, i.e. the model is sensitive to the specific examples, the statistical noise, in the training data (minor variation or statistical  noise on the expected inputs will result in large differences in the  output). This is the result of high variation and low bias.

    <u>Idea</u>: The learning algorithm can be updated to encourage the network toward using small weights. Penalizing a network based on the size of the network weights during training can reduce overfitting. The approach is to add L1 or L2 norm penalties to the loss function to achieve weight decay. 

    * L1: sum of the absolute values of the weights
    * L2: sum of the squared values of the weights 

    <u>Benefits</u>: The addition of norm penalty to the neural network has the effect of reducing generalization error and of allowing the model to pay less attention to less relevant input variables.

    <u>Tips</u>: Standardize input data. If the input variables have different scales, the scale of the weights of the network will vary accordingly. This introduces a problem when using weight regularization because the absolute or squared values of the weights must be added for use in the penalty. This problem can be addressed by either normalizing or standardizing input variables.

  * <u>Early stopping</u>

    <u>Problem</u>: Overfitting of the training data will result in an increase in generalization error, making the model less useful at making predictions on new data. The challenge is to train the network long enough that it is capable of learning the mapping from inputs to outputs, but not training the model so long that it overfits the training data.

    <u>Idea</u>: The general idea is to stop training when the generalization error starts to grow. To achieve this we need

    1. Monitor the model performance on testing data in each iteration. 
    2. <u>Elaborate early-stopping triggers</u>. Most of the time we cannot stop as soon as testing error goes up compared to the testing error in the previous epoch, because training neural network is stochastic and can be noisy. There are some elaborate triggers using some delay or 'patience' strategies:
       * No change in metric over a given number of epochs
       * A decrease in performance observed over a given number of epochs
       * Average change in metric over a given number of epochs
    3. Save the model with the best performance.

  * <u>Data augmentation</u>

    Increasing the size of training data is a way to reduce overfitting. When we are dealing with images, there are a few ways of increasing the size of the training data - rotating the image, flipping, scaling, shifting, etc. 

  * <u>Dropout</u>

    Large weights in a neural network are a sign of a more complex network that has overfit the training data. Probabilistically dropping out neurons in the network is a simple and effective regularization method.

    <u>Idea</u>: Dropping a unit out means temporarily removing it from the network (hidden and visible units), along with all its incoming and outgoing connections. Dropout is not used after training when making a prediction with the fit network.

    <u>Benefits</u>: Dropout has the effect of making the training process noisy, forcing  nodes within a layer to probabilistically take on more or less  responsibility for the inputs.

    <u>Tips</u>: A common value is a probability of 0.5 for retaining the output of each node in a hidden layer and a value close to 1.0, such as 0.8, for retaining inputs from the visible layer. Dropout performs better on larger networks (more layers or more neurons per layer), because large networks can easily overfit the training data.
    
  * <u>Batch-normalization</u> 
  
    <u>Problem</u>: Usually, a dataset is fed into the network in the form of batches where the distribution of the data differs for every batch size. As a result, there might be chances of vanishing gradient or exploding gradient when it tries to backpropagate. In other words, the distribution of the inputs to deep layers can change after each mini-batch when the weights are updated. This change is referred to 'internal covariate shift'.
  
    <u>Idea</u>: Standardize the inputs to a layer for each mini-batch.
  
    <u>Benefits</u>: stabilize the learning process (robust training of deep layers of the network) and dramatically reduce the number of training epochs (Fast training / convergence to the minimum loss function).
  
    <u>Tips</u>: applied to either the activations of a prior layer or inputs directly. Can be used to multiple network types. More appropriate after the activation function if using sigmoid function or tanh function. More appropriate before the activation function if using ReLU, and it's modern default for most network types. Not good to use both dropout and batch normalization in the same network.
  
* **What is an activation function? And discuss the use of an activation function.**

  In mathematical terms, the activation function serves as a gate between the current neuron input and its output, going to the next level. Basically, it decides whether neurons should be activated or not. It is used to introduce non-linearity into a neuron network.

  Activation functions make a linear regression model different from a neural network. We need non-linearity to capture more complex features and model more complex variations that simple linear model cannot capture.

  *  <u>Sigmoid function</u>:  $f(x) = {1\over 1+ \exp(-x)}$

    The output is between 0 and 1. It has some problems such as the gradient vanishing on the extremes. And it's computationally expensive since it uses exponentiation operation.

  * <u>ReLU</u>: $f(x) = \max(0,x)$

    It returns 0 if the input is negative and the value of the input if the input is positive. It solves the problem of vanishing gradient for the positive side, however, the problem is still on the negative side. It is fast because we use a linear function in it.

  * <u>Leaky ReLU</u>: $f(x) = ax, x< 0, f(x) = x, x \geq 0$

    It solves the problem of vanishing gradient on both sides. 

  * <u>Softmax</u>:

    It is usually used at the last layer for a classification problem because it returns a set of probabilities, where the sum of them is 1. 

* **What are the reasons why the loss does not decrease in a few epochs?**

  * The model is underfitting the training data / The regularization hyper parameter is quite large
  * The learning rate is large
  * The initialization is not proper (e.g. initializing all the weights as 0 does not make the network learn any function)
  * The result of vanishing gradient problem

* **Why sigmoid or tanh is not preferred to be used as the activation function in the hidden layer of the neural network?**

  Only a narrow range of input can produce non-zero gradients of sigmoid or tanh. Once the input is out of the range, the function satuates. When sigmoid or tanh activation function saturates, their gradients become close to 0. Once saturated, the algorithm cannot update the weights effectively. So sigmoid and tanh prevent the neural network from learning effectively and result in vanishing gradient problem. 

  The vanishing gradient problem can be addressed by ReLU activation function.

## 2. CNN

* **What is padding?**

  *  <u>Full padding</u>: introduces zeros such that all pixels are visited the same number of times by the filter. Increases size of output.
  * <u>Same padding</u>: ensures that the output has the same size as the input.

* **What is stride?**

  Stride controls how the filter convolves around the input volume. The formula for calculating the output size is
  $$
  O = {W - K + 2P \over S} + 1
  $$
  where $O$ is output dim, $W$ is the input dim, $K$ is the filter size, $P$ is padding and $S$ is the stride.

* **What is CNN?**

  CNNs are composed of layers, but those layers are not fully connected, they have filters, sets of cube-shaped weights, that are applied throughout the image. Each 2D slice of a filter is called kernel. These filters introduce <u>translation invariance</u> and <u>parameter sharing</u>. They are applied through convolution.
  
* **What are the parameters for a convolutional layers?**

  Number of filters, size of kernels, stride, padding, activation function.
  
* **What is pooling?**

  A pooling layer is a new layer added after the Convention layer, specifically, after the nonlinearity (ReLU) has been applied to the feature maps. The pooling layer operates upon each feature map separately to create a new set of the same number of pooled feature maps. Pooling reduces input size of the final fully connected layers.
  
  Pooling involves selecting (but no learning parameters):
  
  * A pooling operation, e.g. max, mean, median
  * The size of the pooling operator (smaller than the size of the feature map, almost 2 * 2 with 2 strides using max pooling)
  * The stride
  
* **What do CNN layers learn?**

  Each CNN layer learns features of increasing complexity.

  * The first layers learn basic feature detection filters: edges, corners, etc
  * The middle layers learn filters that detect parts of objects. (e.g. eyes, noses on faces)
  * The last layers have higher representations: they learn to recognize full objects, in different shapes and positions.

* **How to build a CNN?**

  <u>Convolutional Layers</u>

  1. I/O:
     * Input: previous set of feature maps: 3D cuboid
     * Output: 3D cuboid, one 2D map per filter
  2. Parameters:
     * Number of filters
     * Size of kernels (W and H only, D is defined by input cude)
     * Activation function
     * stride
     * Padding
  3. Action
     * Apply filters to extract features
     * Filters are composed of small kernels, learned
     * One bias per filter
     * Apply activation function on every value of feature map

  <u>Pooling layers</u>

  1. I/O
     * Input: previous set of feature maps, 3D cuboid
     * Output: 3D cuboid, one 2D map per filter, reduced spatial dimensions
  2. Action
     * Reduce dimensionality
     * Extract maximum or average of a region
     * Sliding window approach
  3. parameter
     * Stride
     * Size of window

  <u>Fully connected layers</u>

  1. I/O
     * Input: Flattened previous set of feature maps
     * Output: Probabilities for each class or simply prediction for regression $\hat{y}$
  2. Action
     * Aggregate information from final feature maps
     * Generate final classification, regression segmentation, etc
  3. Parameters
     * Number of nodes
     * Activation function: usually changes depending on role of the layer. If aggregating info, use ReLU. If producing final classification, use Softmax. If regression use linear.

* **Does the regularization techniques still applied to CNN?**

  L1/L2 regularization, data augmentation, early stopping work the same as FFNN. 

  <u>Dropout is less effective at regularizing convolutional layers</u>. Reason 1: conv layers have few parameters and thus need less regularization to begin with. Reason 2: the spatial relationships are encoded in feature maps, thus activations can become highly correlated. 

  Dropout can still be used in dense layers in CNNs (e.g. fully connected layers in VGG), however, modern architecture replaces dense layers with global average pooling to improve performance. So dropout is outdated in modern CNNs.

* **What is the result of backward propagation of maximum pooling that pass to the previous layer?**

  There is no gradient with respect to non maximum values, since changing  them slightly does not affect the output. Further the max is locally linear with slope 1, with respect to the input that actually achieves the max. Thus, the gradient from the next layer is passed back to only that neuron which achieved the max. All other neurons get zero gradient.

## 3. CNN Interpretation

* **What is Layers Receptive Field and why it matters?**

  The <u>receptive field</u> (RF) is defined as the region in the input space that a particular CNN's feature (or activation) is mapping to. The <u>receptive field size</u> of CNN is the size of the region in the input that produces the feature. The size is the key parameter to associate an output feature to an input region, which help us understand the extent to which input signals may affect output features.

  RF can be computed using <u>recursive formula</u> for $L$ layers. ($k_l$ is kernel size, $s_l$ is stride) [[more](https://blog.mlreview.com/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807)]
  $$
  r_0 = 1 + \sum^L_{l=1}\left((k_l-1) \prod^{l-1}_{i=1}s_i\right)
  $$
  The insight is: the closer a pixel to the center of a RF, the more it contributes to the calculation of the output feature. A feature does not only look at a particular RF in the input image, but focus more to the center of that RF.

  Large RFs are necessary for high-level recognition tasks, but with diminishing rewards. 

  To increase the RF in a CNN, we can do: 

  1. add more conv layers (make network deeper); 

     Increase RF size linearly, by the kernel size. But the effective RF size is reducing.

  2. add pooling layers or higher stride convolutions; 

  3. use dilated convolutions; 

     Insert holes between the kernel elements. Dilation rate indicates how much the kernel is widened.

  1) depth-wise convolutions.

* **Dilated CNNs?**

  Dilations introduce holes in a convolutional kernel, by a factor of stride $r$. so the kernel size can be re-defined as $k' = r(k-1)+1$.

  <u>Benefits</u>:

  * Detection of fine-details by processing inputs in higher resolutions.
  * Broader view of the input to capture more contextual information.
  * Faster runtime with less parameters.

* **How to do feature visualization for interpretability?**

  Purposes: 

  1) interpretability and explainability

     Visualize feature space and weight space.

  2) debugging and improving models

     Tensorboard.

  3) teaching deep learning concepts

  Target: 

  1) Computational graph and network architecture

  2) <u>Learned model parameters (weights and filters)</u> 

     * CNN feature extraction visualization

       Extracted features are feature maps (result of filtering), from which we can analyze the patterns. Visualize filters, and the feature maps, to know what features each filter extracts from the input image.

  3) <u>Individual computational units (activations, gradients)</u>

     * Visualize intermediate activations to know the output after each conv operation. Activations are the values of neurons, from which we know what the network saw.

     * Use <u>Tensorboard</u> to explore the output of a specific hidden layers, by visualizing the distribution of gradients, network weights, and activations.
     * Visualize top predictors by input gradients.

  4) <u>Visualize by deconvnets (aka. Deconvolutions/ transposed convolutions)</u>

     To trace back output of each conv layer back to the input image.

  5) <u>Visualize heapmaps (using CAM)</u>

  6) Aggregate information (performance metrics)

* **Why and how to visualize and understand weights?**

  Researchers usually visualize first layer weight; discuss aggregate statistics of weights.

  Main barriers of making sense of weighs:

  * <u>Lack of contextualization</u>: Researchers often visualize first layer weights in vision models, because they are linked to RGB values. Weights in hidden layers are meaningless by default.
  * <u>Dimensionality and scale</u>: NNs have too many neurons.

* **What does each layer in CNN learn and how do we know?**

  The 1st conv layer(s) learn features such as edges and simple textures. The later conv layers learn feature such as more complex textures and patterns. The last conv layers learn features such as objects or parts of objects. The fully connected layers learn to connect the activations from the high-level features to the individual classes to be predicted.

  Since each layer in a convnet learns a collection of filters such that their inputs can be expressed as a combination of these filters. We can visualize filter to get to know what pattern each layer has learned to extract from the input.

  This visualization can provide insight into how the model works, and also helps in diagnosing the model and selecting better architectures (tuning the filter size, stride) for better performance.

* **What is image occlusion method?**

  Occlusion method is a forward pass attribution method, attributing importance for the classification of the image. Occlusion involves running a patch over part of the image to see which pixels affect the classification the most. To obtain fine details we need to use a small occlusion area, increasing the number of model evaluations.

  For every possible patch on the image, we run trained CNN and calculate the difference between the loss without occlusion and the loss with occlusion. We can then get an <u>occlusion loss map</u>.

  The process of occlusion:

  1. Take note of the true label K, input image to a trained network
  2. Get the prediction of the true image, Q
  3. Compute the loss $L_{nol} = -\log P(y=Q)$
  4. Occlude patches of the image with gray blocks starting at the top left.
     * Get the prediction, compute the loss $L_{ocl_i} = - \log P_{occ}(y=Q)$
     * Compute the difference of the losses $L_{nol} - L_{ocl_i}$

  Interpretation: If K=Q, the occlusion method answers what parts of the image have contributed to the correct prediction. If K != Q, the occlusion method answers what parts of the image have contributed to predict the incorrect class.

* **What is Saliency Maps for class model visualization?**

  <u>Saliency maps</u> are a local gradient-based backpropagation interpretation method to measure the spatial support of a particular class in each image. It is frequently used to interpret the predictions of CNNs. Class model visualization method consists of numerically generating an image representing a class. There are five main approaches to get the saliency maps:

  * Gradient based backpropagation
  * Deconvolutional networks
  * Guided back-propagation
  * Class activation maps
  * Grad-CAM and guided Grad-CAM

  For a given class and the score function $S_c(I)$ (the logits of the class c given the image I), we maximize the input image. In other words, we want to find the image that maximizes the logits or any element of the feature map - <u>activation maximization</u>. $I^* = argmax_I S_c(I)$. Sometime, we do L2 regularization $I^* = argmax_I S_c(I) - \lambda ||I||^2_2$, where $S_c(I)$ is the logits of the class c given the image I.

  The resulting images represent local maximum. Since the logits of the network have receptive field as big as the image, we can identify features of the specific class all over the image. Note that we don't use the softmax values, since the maximization of the class can be achieved by minimizing the scores of other classes. 

* **What is Saliency Maps with gradient based backprop for image-specific class saliency visualization?**

  To detect and highlight what part of a specific image $I_0$ (in a specific class) that contains the relevant information, we use backpropagation to compute the gradients of logits with respect to the input of the network while the weights are fixed (for a trained network, the weights were computed with respect to the parameters of the network). We can then highlight pixels of the input image based on the amount of the gradient they receive, showing the contribution to the final score.

  With Taylor expansions, we have $S_c(I) \approx w^T I + b$, where $w = {\partial S_c \over \partial I}|_{I_0}$. Note that for color images, we take the maximum derivative of the three derivatives.

  [[Symonian et al., 2014](https://arxiv.org/pdf/1312.6034.pdf)]

* **What is Deconvolutions and it can be used to examine a convnet?**

  Deconvnet is attached to each of the convnet layers, providing a continuous path back to input image. 

  For example, after training a convnets, all features have been computed. To examine a given convnet activation, we set all other activations in the layer to zero and pass the feature maps as input to the attached Deconvnet layer. Then we do unpool, rectify, filter, etc, to reach the input image. 

  Projection of feature activations back to the input space gives us insights as to what the filters in those layers were activated upon.

* **What is Saliency Maps with DeconvNet?**

  To recognize what features in the input image, that an intermediate layer of the network is looking for, DeconvNet inverts the operations from a particular layer to the input using that specific layer's activation. To invert the process, the authors used:

  1. Unpooling as the inverse of pooling

     They used a set of <u>switch(mask)</u> variables to recover maxima positions in the forward pass, because the pooling operation is non-invertible.

  2. Filtering

     Deconvnet uses transposed versions of the same filters, and applied to the rectified maps.

  3. Inverse ReLU to remove the negative values as the inverse of itself

  [[Zeiler and Fergus, 2013](https://arxiv.org/pdf/1311.2901.pdf)]

* **What is Saliency Maps with guided backpropagation algorithm**

  DeconvNet + Gradient-based backpropagation = Guided backpropagation algorithm (for getting Saliency maps)

  Guided backpropagation algorithm masks the importance of signals based on the positions of both negative values of the input in forward-pass, and the negative values from the reconstruction signal flowing from top to bottom (deconvolution).

  [[Springenberg et al., 2014](https://arxiv.org/pdf/1412.6806.pdf)]

* **What is class activation mapping (CAM)?**

  Different from gradient based methods, which give pixel by pixel importance fro localized feature, CAM gives info about which sub-part of the image the model focuses at when making a particular prediction. CAM is an explanation method for interpreting CNNs. 

  The fully connected layers at the very end of the model are replaced by a layer named <u>Global Average Pooling (GAP)</u>, and combined with a class activation mapping (CAM) technique. GAP averages the activations of each feature map to a single vector. So the number of the resulting vectors is the number of feature maps. Then a weighted sum of the resulted vector is fed to the final softmax layer. After training this CNN, we get $W_i$ from the GAP of the $i$th feature map. For a particular image, we take the weighted sum of the feature maps in the last convolutional layer, and upsample the weighted feature map to the image dimensions to get the output.

  [[Zhou et al., 2016](https://arxiv.org/pdf/1512.04150.pdf)]

* **What is Grad-CAM?**

  <u>Grad-CAM</u> is a more versatile version of CAM that can produce visual explanations for any arbitrary CNN, even if the network contains a stack of fully connected layers as well (e.g. VGG). Grad-CAM is applied to a neural network that is done training - the weights are fixed. We feed an image into the network to calculate the Grad-CAM heatmap for that image for a chosen class of interest. 

  Instead of GAP layer, we compute the average gradients of the logit of class $c$ with respect to the feature maps/ activation maps $k$ of the final conv layer, using $\alpha_k^c = {1\over Z} \sum\limits_{i,j}{\partial y^c\over \partial A_{ij}^k}$, where $\alpha_k^c$ is the importance of feature map k for the class c, $A_{ij}^k$ is the i,j element of the k activation map of the last conv activation layer.

  Instead of using $W_i$, we use $\alpha_k^c$ to calculate the average sum of feature maps, next, use ReLU to activate this weighted sum by $S^c = ReLU(\sum_k \alpha_k^c A^k)$. Then we upsample the weighted feature map to the image dimensions to get the output.

  Grad-CAM can only produce coarse-grained visualization.

* **What is Guided Grad-CAM?**

  <u>Guided Grad-CAM</u> = guided backpropagation + Grad CAM, by performing an element-wise multiplication of guided-backpropagation with Grad CAM.

  They both are local backpropagation-based interpretation methods, model-specific since they can be used exclusively for interpretation of CNNs.

  [[Selvaraju et al., 2016](https://arxiv.org/pdf/1610.02391.pdf)]

* **Limitations of Saliency Maps?**

  * Not always reliable. Subtracting the mean and normalizations can make undesirable changes in saliency maps. Grad-CAM and gradient base are the most reliable (from research).
  * Vulnerable to adversarial attacks
  
* **How to extract the activation maps from intermediate convolutional layers? (Visualize the activations of convolutional layers)**

  Intermediate activations are useful for understanding how successful convolutional layers transform the input, and for getting a first idea of the meaning of individual convnet layers. Note that the output of a layer is often called its activation, so the output of the activation function is the activation map we want to visualize.

## 3. Object Detection and Semantic Segmentation

* *Object detection: classify and locatie*
  * *Sliding window vs. region proposals*
  * *Two stage detectors: the evolution of R-CNN, Fast R-CNN, Faster R-CNN*
  * *Single stage detectors: detection without Region Proposals: YOLO/SSD*
  
* *Semantic Segmentation: classify every pixel*
  * *Fully convolutional networks*
  * *SegNet & U-NET*
  * *Faster R-CNN linked to Semantic Segmentation: Mask R-CNN*

* **What is the difference between image classification and semantic segmentation?** 

  For image classification tasks, we assign a single label to the entire picture. For semantic segmentation, we assign a semantically meaningful label to every pixel in the image. So the output of semantic segmentation is not a class prediction but a picture.

* **Why object detection and semantic segmentation?**

  In computer vision, it's been used for autonomous vehicles, biomedical imaging detecting cancer/diseases, video surveillance (counting people, tracking people), aerial surveillance, geo sensing (tracking wildfire, glaciers, via satellite). In those applications, efficiency, sensitivity, and resolution are important. Real-time segmentation and detection is often needed.

* **What metrics are usually used to measure accuracy?**

  * <u>Pixel accuracy</u>: percent of pixels that are classified correctly
  * <u>IOU</u>: Intersection-Over-Union (Jaccard index): Overlap / Union
  * <u>mAP</u>: Mean Average Precision: AUC of Precision-Recall curve standard (0.5 is high)
  * <u>DICE</u>: Coefficient (F1 Score): 2 * Overlap / Total number of pixels

* **How to do object detection by region proposals?**

  Object detection is combined classification and localization - <u>multi-task learning</u>. First, do classification using standard CNN. Second, do localization using regression for predicting box coordinates. Last, combine loss from classification (Softmax loss) and regression (L2-norm loss).

  For multiple objects in a single image, we apply <u>region proposals methods</u> to find object-like regions. For example, <u>selective search algorithm</u> returns boxes that are likely to contain objects (no classification yet). It starts from small superpixels, using hierarchical segmentation, merge pixels based on similarity.

  Then do <u>Region-based CNN (R-CNN)</u> for classification. In slow R-CNN, each region is forward through a CNN, which is slow. While in Fast R-CNN, region proposal is performed on the output feature of the convnet, instead of the original image. Next, the regions from the features are cropped and resized, and fed in then next convolution layer. Then, the regression loss and softmax loss are calculated, coordinates and the classes are determined. Faster R-CNN is faster than slow R-CNN.

  However, Faster R-CNN can do even faster than Fast R-CNN. In Faster R-CNN, CNN makes proposals -  <u>CNN Region Proposal Network (RPN)</u> predicts region proposals from feature maps. RPN works to classify object / not object, and regress box coordinates. Another CNN layer works to make final classification (report scores), and final box coordinates. In total, there are four losses.

* **What is the single-stage detection without region proposals?**

  Make a grid on the image, within each grid, create a set of base boxes $B$ centered at each grid cell, regress over each base box, make class predictions.

  YOLOv3.

* **What is fully convolutional networks (FCN) for semantic segmentation?**

  FCN is a network with a bunch of conv layers to make predictions for all pixels all at once. It contains a encode (localization) and a decoder (segmentation). The encoder works to downsample through convolutions and pooling, reduce number of parameters. The decoder works to upsample through transposed convolutions, batchnormalization, ReLU. The output of decoder goes through a softmax. The loss of FCN is cross-entropy loss on every pixel. 

  Pros of FCN: 

  * popularize the use of end-to-end CNNs for semantic segmentations
  * Re-purpose imagenet pretrained networks for segmentation = Transfer learning
  * Upsample using transposed layers

  Cons of FCN: 

  * Upsampling means loss of information during pooling.

* **What is U-NET and how it works?**

  U-NET consists of a symmetric encoder and decoder. 

  * <u>Skip connection</u>:  it has skip connections from the output of convolutions blocks to the corresponding input of the transposed-convolution block at the same level: these skip connections allow gradients to flow more effectively and provides information from multiple scales of the image.
  
  * <u>Location information</u> from the down sampling path of the <u>encoder</u>
  * <u>Contextual information</u> in the upsampling path by the <u>concatenating long-skip connection</u>

## 4. SOTA Networks and Transfer Learning

* **What are the two ideas proposed in GoogLeNet?**

  1. <u>Inception blocks</u>: The motivation behind inception networks is to use more than a single type of convolution layer at each layer. In inception layer, it uses kernels with different sizes in convolutional layers and max pooling layers in parallel. Besides, it uses 1 * 1 convolutions to reduce the channel dimension between hidden conv layers. The inception network is formed by concatenating other inception modules. It includes several softmax output units to enforce regularization.

  2. <u>Auxiliary block</u>: In GoogLeNet, there are two auxiliary blocks, one attached to the second inception block and the other to the fifth inception block. It works to take the output, predict the class, and calculate the loss. So the network has two more loss functions. This is beneficial, because when the NN is deep, the gradients are hard to go back to the first several layers to update weights, but with auxiliary blocks, gradients can go back early.

     The total loss is $L = L_{end} + L_1 + L_2$, the gradient is calculated as $\nabla L = \nabla (L_{end} + L_1 + L_2)$. Since the loss functions are independent, when calculating the gradient wrt the layer after both auxiliary blocks, 

* **What are the ideas proposed in MobileNet?**

  <u>Depth-wise separable convolution (DW)</u>:  it combines a depth wise convolution and a pointwise convolution.

* **What are the ideas proposed in DenseNets?**

  It allows maximum information (and gradient) flow by connecting every layer directly with each other. It exploits the potential of the network through feature reuse. 

  There are dense blocks in the hidden layers. Between dense blocks is a transition layer containing batch normalization, 1*1 convolutional layer, and average pooling.

* **Lists some other SOTA networks**

  MobileNetV2, Inception-Resnet (v1 and v2), Wide-Resnet, Xception, ResNeXt, ShuffleNet (v1 and v2), Squeeze and Excitation Nets.

  [[The evolution of image classification explained](https://stanford.edu/~shervine/blog/evolution-image-classification-explained)]

* **What is transfer learning?**

  Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on the second task. It is a popular approach in deep learning where pre-trained models are used as the starting point for computer vision and natural language processing tasks, because these tasks require vast computing and time resources to develop neural network models.


* **When is transfer learning useful?**

  Transfer learning is used for tasks where is data is too little to train a full-scale model from the beginning. 

  In models for computer vision tasks, the initial layers have a common behavior of detecting edges, then a little more complex but still abstract features, and so on. Therefore, a pre-trained models' initial layers can be used directly. 

* **What is representation learning in transfer learning?** [revise needed]

  Representation learning uses big net to extract features from new samples, which are then fed to a new classifier.

  We freeze the weights in trained convolutional layers, fine tune some later layers, and remove fully connected layers. We have to be careful not to have big gradient updates.

  The earlier layers learn highly generic feature maps (e.g. edges, colors, textures). Later layers learn abstract concepts (e.g. dog's ear). 
