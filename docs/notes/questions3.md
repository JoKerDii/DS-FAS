# Machine Learning Concept Q&A (IV)

## 1. NLP

* **What are the tokenization approaches?**

  As a foremost step, we need to <u>split the text into smaller units or *tokens*</u>, since the State of The Art (SOTA) DL architectures (RNNs, GRU, LSTM, transformer based models) in NLP process the raw text at the token level. The entire set of tokens is called *vocabulary*. The size of vocabulary decides the size of training set. There is an area of research on tokenization. Two of the approaches:

  * <u>Word tokenization</u>

    Split by a certain delimiter (e.g. whitespace). Pretrained Word Embeddings (e.g. Word2Vec and GloVe comes under word tokenization)

    Drawbacks: 

    * deal with *Out of Vocabulary (OOV)* words by replacing the rare words in training data with *unknown tokens (UNK)*, but information is lost as we map OOV to UNK which are of the same representation.
    * Large memory required.

  * <u>Sub-word tokenization</u>

    Split on statistically significant fragments. More flexible but loss interpretability. Transformer based models rely on it to prepare vocabulary. Example: *Byte Pair Encoding (BPE)*.

    * BPE is a word segmentation algorithm that merges the most frequent character (sequences) iteratively. 
    * BPE address the issues of Word Tokenizers, tackles OOV effectively at test time. The BPE learned operation merges the characters into larger known symbols.

* **What is a language model and how to evaluate a language model?**

  <u>A *language model* is a probability distribution over words or word sequences. It estimates the probability of any sequence of words</u>. The applications of language model is: text recognition, speech recognition, text to speech, sentence prediction, translation, named entity recognition.

  Evaluation methods:

  * Base approach (extrinsic)

    Transform it into a <u>classification problem</u>.

  * Intrinsic evaluation

    Train model on training data and measure the quality on test data, by a prob-based metric - *perplexity*.

  * Formal metric: <u>perplexity (PP)</u>

    <u>In NLP, perplexity is a way of evaluating language models.</u> <u>Perplexity is the inverse probability of the test set, normalized by the number of words. It measures how confused or perplexed a model is against an unseen sample.</u> <u>The best language model is the one that best predicts an unseen dataset, so the smaller perplexity indicates a better model.</u>

    Given a sentence of N words $W = (w_1, ..., w_N)$, the language model gives the probability of the sentence $P(w_1, ..., w_N)$, then the perplexity of a sentence is $PP(W) = P(w_1, ..., w_N)^{-1/N}$. The higher probability for a sentence means lower PP. A low PP score means the model learned.

    Suppose a bad model predicts random words, the probabilities of words are the same $P(w_i) = {1\over |Vocabulary|} = {1\over V}$, the probability of the sentence is $P(w_1, ..., w_N) = P(w_1)P(w_2) ... P(w_N) = {1\over V} {1\over V} ... {1\over V} = ({1\over V})^N$, then $PP(W)=P(w_1, ..., w_N)^{-1/N} = (({1\over V})^N)^{-1/N} = V$.

* **How to build language models?**

  * Naive approach: <u>Unigram model</u>

    Assume each word is independent of all others, count the frequency of each word in the training data. The probability of each word is $P(w_i) = {n_{w_i}(d)\over |W|}$, where $n_{w_i}$ is the number of times a word $w_i$ appears in the corpus, $|W|$ is the total number of words in corpus. The the probability of a sentence is $P(X) = P(w_1) ... P(w_N)$.

    Drawbacks: Context does not play a role.  

    To deal with OOV in test data, we apply <u>smoothing</u>  to move a little bit prob mass from seen to unseen. One type of smoothing is called <u>add-$\alpha$ smoothing:</u> $P(W) = {n_W(d) \over |W|} = {n_w(d) + \alpha \over |W| + \alpha |V|}$, where $|V|$ is the number of unique words in the vocabulary (including 1 for unknown words \<UNK\>), $\alpha$ are usually small 0.5-2. Whenever a word $W$ is not found in the vocabulary it is replaced with a token \<UNK\> representing unknown.

  * Easiest Approach: <u>Bigram model</u>

    Bigram model looks at pairs of consecutive words. Calculate the probability of a word given all previous words by using only the conditional prob of one preceding word $P(w_2|w_1) = {P(w_1, w_2)\over P(w_1)}$. The assumption is the prob of a word only depends on the previous word - *Markov assumption*.

    A bigram prob of  a word $w_n$ given a previous word $w_{n-1}$ is computed by $P(w_n | w_{n-1}) = {C(w_{n-1}w_n)\over \sum_i C(w_{n-1}i)} = {C(w_{n-1}w_n)\over  C(w_{n-1})} $, where $C(w_{n-1}w_n)$ is the count of the bigram, and the denominator is the sum of all the bigrams that share the same first word, i.e. the unigram count of that word.

    Drawbacks: No semantic information conveyed by counts (e.g. vehicle vs car). 

  * General: <u>N-gram model</u>

    In practice, when there are enough training data, it's more common to use trigram or 4-gram models.

    To estimate bigram and ngram probabilities, we apply *Maximum Likelihood Estimation (MLE)* approach.

    Drawbacks: 

    * Feature space represented by n-gram models is extreme sparse. Sparse data is the problem with MLE for estimating probability function.
    * Model depends on the training corpus and the number N.

  * <u>TF-IDF</u>

    *Term frequency (TF)* measures the frequency of a word in a document. This highly depends on the length of the doc and the generality of word. The solution is to normalize the count value of each word with the total number words in the doc. $TF(t,d) = {\text{count of t in d} \over \text{number of words in d}}$, where $t$ is the word/term, $d$ is the sentence/document.

    *Document frequency (DF)* measures the importance of the word t across all documents in the corpus. DF is needed because a common word through normalization will still have a high importance in a doc. $DF(t) = {\text{occurrence of docs with t}\over \text{number of docs}}$.

    *Inverse document frequency (IDF)* measures the informativeness of the word t. $IDF(t) = \log ({1\over DF(t)+1})$. As the size of the corpus increases, DF explodes, hence the log is used.

    TF-IDF score: $\text{TF-IDF}(t,d) = TF(t,d) \log ({1\over DF(t) + 1})$.

* **How do neural networks work for language modeling?**

  In NN, each word is represented by a *word embedding*. Words that are more semantically similar to one another will have embeddings that are also proportionally similar. Word2Vec and GloVe are pre-existing word embeddings that have been trained on gigantic corpora.

  Feed-forward neural network for language modeling:

  Strength:

  * No sparsity issues
  * No storage issues

  Drawback:

  * Fixed-window size can never be big enough but we need more context
  * No concept of time

* **What are the advantages of RNNs?**

  * Handle **variable-length** sequences

  * Keep track of **long-term** dependencies

    RNNs remember things learnt from prior inputs while generating outputs.

  * Maintain information about the **order** as opposed to FFNN

  * **Share parameters** across the network

    The success of CNNs and RNNs can be attributed to the concept of parameter sharing, which is an effective way of leveraging the relationship between one input item and its surrounding neighbors.

* **How does a simple RNN (one layer) work?**

  The RNN has loops for information to persist over time, that's why it's called 'recurrent' NN. At each time step, the RNN is fed the current input and the previous hidden state. 

  The hidden state given by <u>recurrence relation</u>: $h_t = f_{u,v}(h_{t-1},x_t)$, where $h_t$ is state/hidden state/encoding/embedding at time $t$. $f_{u,v}$ is a function parameterized by $u,v$ (bias ignored), which are learned during training. $x_t$ is a input vector at time step $t$.

  Given a input vector $X_t$, at each time step $t$, the RNNs are trained by updating hidden state $h_t = \tanh (U h_{t-1} + VX_t + \beta_1)$ ($\beta_1$ is for calculating hidden states, there is no bias term for $VX_t$), and generating output $Y_t = \sigma (Wh_t + \beta_2)$ (Sigmoid used for binary classification). $U,V,W$ are three types of weights matrices learned during training. 

  Loss is calculated at each time $t$, as a function of $Y_t, \widehat{Y_t}$. The losses are aggregated by $L = \sum^N_{t=1}L_t$, where $N$ is the number of word embeddings or the number of time points.

  The RNN above is a one-layer simple RNN. But RNN can be deep by arranging units in layers. The output of each unit is the input to the other units. 

* **What is back-propagation through time (BPTT) in RNN and What is the back-propagation issue in RNN?**

  BPTT is to do back propagation on an unrolled RNN. Since the error of a given time step depends on the previous time step, the error is back propagated from the last to the first time step. This requires to calculate loss for each time step and allows updating the weights. Note that BPTT can be computationally expensive with a high number of time steps.

  For longer sentences, back-propagating through more time steps requires the gradients to be multiplied many times which causes the following issues.

  * <u>Vanishing gradient problem</u>: If many values < 1, then the gradient will be close to zero. This causes the parameters to update very slowly. This can be solved by LSTM.
  * <u>Exploding gradient problem</u>: If many values > 1, then the gradient will explode. This causes an overflow problem. This can be solved by truncating or squashing the gradients.

* **What is Bidirectional RNNs?**

  Bidirectional RNNs process sequences in both directions, forward direction and reverse direction.

  * Hidden state of forward RNN: $h_t^F = \tanh(V^FX_t + U^F h_{t-1}^F + \beta_1^F)$
  * Hidden state of reverse RNN: $h_t^R = \tanh(V^R X_t + U^R h_{t-1}^R + \beta_1^R)$
  * Final hidden state: $h_t^T = h_t^F + h_t^R$

  The output is $\widehat{y_t} = \sigma(h_t^T W + \beta_2)$.

* **What are the flavors of RNNs?**

  * One to One (useless)

  * One to Many

  * Many to One (for sentiment analysis; classification)

  * Many to Many (for language modeling; machine translation)

    Many to many inference for language modeling, where during inference the output of one unit is given as the input to the next unit.

* **What are the shortcomings of RNN?**

* **What is Gated Recurrent Unit (GRU)?**

  $\tilde{h_t} = \tanh (VX_t + U[R_t \bigodot h_{t-1}] + \beta_1)$

  $h_t = Z_t \bigodot h_{t-1} + (1-Z_t) \bigodot \tilde{h_t}$

  Reset Gate: $R_t = \sigma(V_R X_t + U_R h_{t-1} + \beta_R)$. The update gate decides whether the cell state should be updated.

  Update Gate: $Z_t = \sigma(V_Z X_t + U_Z h_{t-1} + \beta_Z)$. The reset gate decides whether the previous cell state is important.

  Strength:

  * Current input can affect how much of the past information to consider
  * The update gate solves the vanishing gradient problem
  * Hidden state more robust to outlier inputs because of the update gate

  Drawback:

  * The same hidden state is used for memory and output
  * With only two gates, performance suffers on longer sequences

  [[d2l.ai GRU](https://d2l.ai/chapter_recurrent-modern/gru.html)]

* **What is Long Short Term Memory (LSTM)?**

  $\tilde{c_t} = \tanh(VX_t + Uh_{t-1} + \beta_1)$

  $c_t = f_t \bigodot c_{t-1} + i_t \bigodot \tilde{c_t}$

  $h_t = o_t \bigodot \tanh (c_t)$

  Forget Gate: $f_t = \sigma(V_f X_t + U_f h_{t-1} + \beta_f)$.  Forget gate controls whether the previous states should be forgot. 

  Input Gate: $i_t = \sigma(V_i X_t + U_i h_{t-1} + \beta_i)$

  Output Gate: $o_t = \sigma(V_o X_t + U_x h_{t-1} + \beta_o)$. The output gate controls which part of the cell outputs to the hidden state.

  LSTM is capable of learning long-term dependencies by introducing changes to how we compute outputs and hidden state using the inputs. 

  [[Great blog for LSTM](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)] [[parameters in LSTM](https://medium.com/deep-learning-with-keras/lstm-understanding-the-number-of-parameters-c4e087575756)]

* **What are Word Embeddings? Why we prefer Word Embeddings?**

  Word Embeddings are a numerical vector representation of the text in the corpus that maps each word in the corpus vocabulary to a set of real valued vectors in a pre-defined N-dimensional space. 

  They are learned by NN models on tasks (e.g. sentiment analysis, document classification) or by unsupervised techniques (e.g. statistical analysis of documents). Embeddings learned from large-scale NLP models can  be extremely effective representations for <u>semisupervised transfer learning</u>.

  Reasons why we prefer word embeddings.

  * Disadvantage of the simplest way to represent words: One-hot encoding.

    * <u>Scalability issue</u>. The dimension of OHE vectors could be infinitely large in real world data
    * <u>Sparsity issue</u>. The OHE vectors have 1 for a single value and 0s everywhere else. It's hard for models to learn, so the model will not generalize well over the test data.
    * <u>No context captured</u>. Lose contextual and semantic information by blindly creating vectors.

  * Other simple methods: e.g. TF-IDF, face one or more issues mentioned above.

  * Word Embeddings resolve all issues above.

    The vectors are learned such that they capture the shared context and dependencies among the words. And the model can learn better and generalize well.

* **Introduce some of the Word Embedding methods?**

  * <u>Word2Vec</u>

    Word2Vec uses distributional hypothesis (words occurring in similar linguistic contexts will have similar semantic meaning) to map words having similar semantic meaning geometrically close to each other in a N-dimensional vector space. 

    It can be implemented using either Common Bag of Words (CBOW) or Skip Gram.

  * <u>Glove</u>

    While Word2Vec relies on local statistics to derive local semantics or words, Glove combines local with global statistics such as Latent Semantic Analysis to capture the global semantic relationships of a word.

  * <u>ELMo</u>

    Glove and Word2Vec fail to distinguish between the polysemous (having different meaning and senses) words. ELMo resolves this issue by taking in the whole sentence as an input rather than a particular word and generating unique ELMo vectors for the same word used in different contextual sentences. ELMo uses <u>2-layer Bi-directional LSTM</u> to produce word vectors.  

    Properties:

    * Contextual
    * Character based

    Applications:

    * Improve document search and information retrievals
    * Improve language translation system (e.g. fastText, Seq2Seq)
    * Improve text classification accuracy (e.g. sentiment analysis, spam detection, document classification)

  * <u>BERT</u>

  * <u>GPT</u>

* **How does ELMo work?**

  First ELMo convert each token to an appropriate representation using <u>character embeddings</u>. The character embedding representation is then run through a convolutional layer using some number of filters, followed by a max-pool layer. Finally this representation is passed through a 2-layer <u>highway network</u> before being provided as the input to the LSTM layer. The advantages of this transformation:

  * Using character embeddings allows us to pick up on morphological features that word-level embeddings could miss. 
  * Using character embeddings ensures to form a valid representation even for out-of-vocabulary words.
  * Using convolutional filters allows to pick up on n-gram features that build more powerful representations.
  * The highway network layers allow for smoother information transfer through the input.
  
  The ELMo runs <u>separate multi-layer forward and backward LSTMs</u> and then concatenates the representations at each layer. (It's NOT running a layer of forward and backward LSTM, concatenating, and then feeding into the next layer).  A <u>residual connection</u> is added between the first and second LSTM layers, the input to the first layer is added to its output before being passed on as the input to the second layer. The advantage of this structure:
  
  * Residual connection is important for training, since the ELMo language model is large-scale (LSMTM layers have 4096 units, conv layers uses 2048 filters). 
  * Fine tuning the language model on task-specific data leads to drops in perplexity and increases in downstream tasks performance.

  The math details of EMO performing operation on work $k$: $ELMo_k^{task} = \gamma_k(s_0^{task} \cdot x_k + s_1^{task} \cdot h_{1,k} + s_2^{task} \cdot h_{2,k})$, where $s_i$ is softmax-normalized weights on the hidden representations from the language model and $\gamma_k$ is a task-specific scaling factor.

  * How it's trained? The ELMo LM is built on a sizable dataset: 1B Word Benchmark. We learn a separate ELMo representation for each task (e.g. sentiment analysis) the model is used for by fine tuning. $\gamma_k$ and $s_i$ are learned during training of the task-specific model. 
  * How it's used? To use ELMo, we freeze the weights of the trained language model and then concatenate ELMo$_k^{task}$ for each token to the input representation of each task-specific model. 
  
* **What is embedding layer in NN?**

  Embedding layer transforms every word into a fixed length vector with arbitrary dimension. E.g. one-hot-encoding or word2vec.

* **What is skip connection?**

  https://www.analyticsvidhya.com/blog/2021/08/all-you-need-to-know-about-skip-connections/

* **Remove stop words, word stemming, word lemmatization**

  *Word stemming* is the process of transforming a word into its root form that allows us to map related words to the same stem. Stemming removes word endings to group together words with the same stem.

  *Word lemmatization* looks up the part of speech of a word and converts it to a noun form.

* **What is Sequence-to-Sequence (seq2seq) and how it works?**

  <u>Encoder Decoder Architecture</u>: Seq2seq models are comprised of 2 LSTM models (or 2 GRU models): 1 encoder, 1 decoder. The final hidden state of the encoder RNN is the initial state of the decoder RNN.

  <u>Encoder</u>: 

  * Context vector $c_i$ and hidden state $h_i$ are two internal states maintained by LSTM at every time step $i$. 
    * *Context vector* or *internal state vector* is used to encapsulate the information of input. 
    * Hidden states are calculated by $h_t = f(W^{hh}h_{t-1} + W^{hx}x_t)$. 
  * The output of encoder is $y_i$ for each time step $i$.  They are probability distributions over the entire vocabulary generated by a softmax activation.

  <u>Decoder</u>: 

  * The initial states are initialized to the final state of the encoder (i.e. the context vector of the encoder's final cell).  The decoder is trained to start generating the output depending on the information encoded by the encoder.
  * Each unit accepts previous hidden state $h_{i-1}$ and produces output $y_i$ and its own hidden state $h_i$.
    * $h_t = f(W^{hh}h_{t-1})$
    * $y_t = softmax(W^S h_t)$
  * The loss is calculated on the predicted outputs from each time step and the errors are backpropagated through time in order to update the parameters of the network.

* **What are the drawbacks of seq2seq models?**

  * <u>Fixed encoder representation</u>: The final hidden state of the LSTM encoder is a fixed dimensionality vector. The more we try to force to compress into a single vector, the more the NN loses, because encoder is doing compression. For encoder, it is hard to compress the sentence. For decoder, different information may be relevant at different steps. The problem is addressed by <u>attention</u>.
  * <u>Vanishing gradients</u>. The longer the sequence is, the deeper the NN is along the time dimension, the harder it is to train. This results in vanishing gradients, where the gradient signal from the objective disappears through backpropagation. 

* **What is attention and how it works to improve seq2seq?**

  An attention mechanism is a part of NN, it makes a model focus on different part of the input. At each decoder step, it decides which source parts are more important. The main idea is that a network can learn which parts are more important at each step, since every parts are differentiable.

  With attention, encoder does not need to do compression, instead, it gives representations for all input tokens. 

  * At each decoder step $h_t$, the decoder receives attention input which are all encoder states $s_1, s_2, ..., s_m$, 
  * Compute <u>attention scores</u> by an <u>attention function</u> for each encoder state (scores$(s_k, h_t), k = 1,...,m$: relevance of source token with this decoder state), 
  * Compute a probability distribution by softamx function as <u>attention weights</u> $a_k^{(t)} = {\exp(score(h_t, s_i))\over \sum^m_{i=1}\exp(score(h_t, s_i))}, k=1,...,m$, and 
  * Compute the weighted sum of encoder states with attention weights as <u>attention output</u> $c^{(t)} = a_1^{(t)}s_1 + a_2^{(t)}s_2+ ... + a_m^{(t)}s_m = \sum^m_{k=1}a_k^{(t)}s_k$. 

* **What is Transformer and how it works?** [[link](https://www.numpyninja.com/post/transformers-understanding)]

  Transformer is solely based on attention mechanisms without recurrence or convolutions. <u>Self-attention</u> is the key part of transformer. In the encoder, self attention looks from each state of a set of states, at all other states in the same set. (While decoder-encoder attention looks from one current decoder state at all encoder states). In the decoder, masked self attention is used to mask out future tokens so that decoder will not look ahead.

  Encoder: all source tokens are repeatedly doing

  * Look at each other (self attention)
  * Update representation

  Decoder: target token at the current step is repeatedly doing

  * Look at previous target tokens (masked self attention)
  * Look at source representations (at the encoder states)
  * Update representation

* **What are the strength of Transformer?** [[link](https://data-science-blog.com/blog/2021/04/22/positional-encoding-residual-connections-padding-masks-all-the-details-of-transformer-model/)]

  RNNs are not feasible for remembering the context in very long sequences, because when the sequence is very long, <u>vanishing gradient problem</u> happens. 

  * <u>Self attention</u>

    Self attention minimizes maximum path length between any two input and output positions in network composed of the different layer types. The shorter the path between any combination of positions in the input and output sequences, the easier to learn long-range dependencies.

  * <u>Positional encoding</u>

    Multi-head attention mechanism does not explicitly uses the the information of the orders or position of input data. Instead, but uses the information implicitly by positional encoding.

    To use the sequence order information, we can inject absolute or relative positional information by adding positional encoding to the input representations (in encoder). Positional encodings can be either learned or fixed. Fixed positional encoding can be done by using sine and cosine functions.

  * <u>Residual connection</u>

    If we simply stack NN, we would suffer vanishing gradient problem during training. Residual connections make a bypass of propagation and so help the network to train by long gradients to flow through the networks directly.

* **What is BERT and how it works?** [[link](https://huggingface.co/blog/bert-101)]

  * Bi-directional encoder representations from transformers (BERT) is a trained model on <u>large informational training dataset</u> that can be used on a wide range of NLP tasks by applying transfer learning. 

  * <u>Masked language model (MLM)</u> enables bidirectional learning from text by masking (hiding) a word in a sentence and forcing BERT to bidirectionally use the words on either side of the covered word to predict the masked word. <u>Next sentence prediction (NSP)</u> helps BERT  learn relationship between sentences. BERT is trained on both MLM (50%) and NSP (50%) at the same time.
  * <u>Transformer</u> parallelizes training BERT. Transformer with attention mechanism helps BERT learn relationships between words. 

* **What are the issues of BERT?**

  * The vanilla BERT has 109482240 trainable parameters. It can only trained by large corporations with massive resources.
  * The sheer size also makes it extremely slow to train
  * The fine-tuning is not straight-forward and requires lots of tweaks and experimentation (for example, we have to use the same tokenizer used during pre-training)

* **Why Transformer is better than RNN/LSTM?**

  Transformer has three characteristics. 

  * <u>Non-sequential</u>, meaning that sentences are processed as a whole in Transformer rather than replying on past hidden states to capture dependencies with previous words. This feature allows Transformer to not suffer from long dependency issues as RNN/LSTM.
  * <u>Self-attention</u>. It is used to compute similarity scores between words in a sentence. It provides information about the relationship between different words
  * <u>Positional embeddings</u>. It uses fixed or learned weights which encode information related to a specific position of a token in a sentence. It provides information about the relationship between different words.

  In summary, RNN suffers from vanishing gradients, LSTM was trying to solve this issue but still relying on gate/memory mechanisms to pass information from old steps to the current one. Transformer architecture leaves no room for information loss. On top of that, Transformer can look at both past and future elements at the same time. The only drawback of Transformer is that processing data through self-attention takes $O(N^2)$ time.
