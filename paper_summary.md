# Basic Word embeddings
<details>
<summary>
<a href="http://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf">NLP (Almost) from Scratch(2011)</a>
</summary>

- Contributions
  - Neural-Net based general framework which is applicable to variety of NLP tasks
    - POS tagging, chunking, NER, Semantic role labeling, Parsing, Anaphora resolution, word-sense disambiguation, ...

- Fully-supervised approach
  - Features for words are extracted
  - Embed the words to a latent vector space(word feature space).
  - Learn higher level features from the latent vector inputs.
  - Train to maximize the log-likelihood of training corpus

- Semi-supervised approach
  - Learn language models from a large-scale unlabeled data
  - Transfer Learning: Initialize the the word embeddings for supervised learning, using the parameters learned from unsupervised language models.

- Reference: 
</details>
<details>
<summary>
<a href="http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf">Distributed Representations of Words and Phrases and their Compositionality(2013)</a>
</summary>

- Contribution
  - Efficient learning algorthm for word embedding tasks, using
    - Skip-gram
    - Negative Sampling
    - Subsampling
  - Good performance in semantic & syntatic analogy tasks

- How it works
  - Negative Sampling: Learn word embeddings so that the embeddings can be used to discriminate the words from the training corpus against the noise words.
  - Subsampling: during training, ignore frequent words by a certain probability(somewhat similar to dropouts)

- Notes
  - In general, CBOW model is faster to train.
  - In general, Skip-gram model shows decent performance for all tasks, relative to the CBOW model.

- Limitations
  - One word, one embedding -> polysemy, or global word context is ignored
  - Cannot utilize the morphological information well: car != cars
</details>



# Multilingual word embedding
<details>
<summary>Mutilingual word embedding in general</summary>

- Motivation for Multilingual word embedding
  - Goal: Learns word embeddings that can generalize over *different languages* and *different  NLP tasks*
  - Why it is needed: For Transfer Learning. Labeled text data is abundant for only few languages(e.g. English), so it may be a good idea to transfer the representations from the languages with abundant labeled text data to the languages with scarce text data for a NLP task.
</details>

<details>
<summary>
<a href="https://arxiv.org/abs/1410.2455">Bilbowa: Fast bilingual distributed representations without word alignments(2014)</a>
</summary>

- Previous Approaches
  - Offline Alignment: Learn word embeddings for each languages separately, and learn the projection from one language to the other. The quality of a generalized projection is questionable.
  - Parallel-Only: Train word embeddings simultaneously for both languages, using only the training sentences that can be aligned in parallel. Only utilizes limited amount of data.
  - Jointly-Trained: Train monolingual word embeddings with cross-lingual penalized loss function. Slow to train.
- How it works
  - Takes the Jointly Trained Approach, with the following modifications:
    - Replaces softmax objective with the noise-contrastive objective(as in word2vec)
    - Sampling algorithm to approximate the cross-lingual loss
- Contributions
  - Enhances the training speed using the Jointly-Trained Approach
- Limitations
  - Still slow
  - Tested only on English-German, English-Spanish pairs
</details>

<details>
<summary>
<a href="https://papers.nips.cc/paper/5270-an-autoencoder-approach-to-learning-bilingual-word-representations.pdf">An Autoencoder Approach to Learning Bilingual Word Representations(2014)</a>
</summary>

- How it works
  - Uses Autoencoder to obtain latent representation of bilingual representation of words
  - A bag-of-words in language $A$ will be reconstructed to the corresponding bag-of-words in another language $B$
- vs Bilbowa
  - Nonlinear transformation of inputs
- Limitations
  - Evaluated on the limited dataset.(EN-German)
</details>

# Language Modeling
<details>
<summary>
<a href="http://aclweb.org/anthology/P96-1041">An Empirical Study of Smoothing Techniques for Language Modeling(1996)</a>
</summary>

- Motivation: Limitations of traditional discrete n-gram based Language Models
  - n-gram based discrete word representation suffers from the **curse of dimensionality**
    - $\left| V \right|$ is often very large, so word counts are often very sparse with many zero entries
    - Too many parameters(probabilities) to learn, because of the high dimensionality
    - Generalizes poorly: a small change in a discrete variable may result in a dramatic impact on the estimation.
  - Smoothing techniques mitigate the problem to a certain degree.
- Contribution
  - Empirical Comparison between various smoothing methods for **n-gram based discrete, count-based statistical language models**
  - Proposes two new smoothing methods that works quite well
- Summary
  - Katz smoothing, Jelinek-Mercer Smoothing works well
  - New methods proposed in the paper(one-count, average-count) also works well
</details>

<details>
<summary>
<a href="http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf">A Neural Probabilistic Language Model(2003)</a>
</summary>

- Contributions
  - Proposes **distributed representation of words** to tackle the **curse of dimensionality** for discrete n-gram based language models
  - Learns both the pdf for word distributions and the distributed representations of words in linear time.
- How it works
  - Represent each words by **latent feature vectors**, so that similar words can have similar representations.
  - **Use Neural Networks to map the feature vectors to logits**, and output the word probabilities by letting the logits pass through another softmax layer.
    $$
    \begin{eqnarray}
    &\hat{P}(w_t|w_{t-1},...,w_{t-n+1})=\frac{e^{y_{w_t}}}{\sum_{i=1}^{|V|}{e^{y_i}}}\\
    &y=nn(v(w_{t-1}),...,v(w_{t-n+1});\theta)\in\mathbb{R}^{|V|}\\
    &y_i:\text{logit of the i-th word, given }w_{t-1},...,w_{t-n+1}\\
    &nn:\text{a neural network}\\
    &v(w_t):\text{distributed representation of }w_t
    \end{eqnarray}
    $$
  - Optimizes the training corpus log-likelihood using SGD
- Limitations
  - Slow to train(at 2003)
  - Context length should be fixed
</details>

# Contextual Language Model
<details>
<summary>
<a href="https://arxiv.org/pdf/1511.03962.pdf">Document Context Language Models(2015)</a>
</summary>

- Problems with Previous Language Models
  - Neural Language Models(Bengio et al. 2003): supports only fixed-length context
  - Recurrent Neural Network Language Models(2010)
    - pros: Possible to condition on contexts of arbitrary length
    - cons
      - **Document-level context is ignored**: Context scope is limited to sentence-level, so information from the previous sentences are ignored.
      - Problems with learning document-level context in conventional RNNLMs
        - Conventonal RNNLMs learn Language Models on documents by regarding the whole document as a single sentence.
        - Information decay:  In this case, meaningful document-level information fail to survive for a long time.
        - Difficulty in learning: Since the whole document is feeded as an input, RNNLMs should deal with with many time steps when learning.
- Document Context Language Models
  - Contributions
    - Incorporates the information from a previous setence as a context to the following sentence.
    - 'Short-circuit' approach: Feed the information from a previous sentence **directly** to the LSTM layers that processes the current sentence.
</details>

<details>
<summary>
<a href="http://proceedings.mlr.press/v37/jernite15.pdf">A Fast Variational Approach for Learning Markov Random Field Language Models(2015)</a>
</summary>

- Contributions
  - Proposes an Efficient Variational Inference Algorithm for MRF-based Language Models
- vs Neural Language Models
  - MRF LM optimizes the **global** data likelihood, given the order $K$ for the Markov Sequence Model. In contrast, Neural LMs try to extract one token at a time, and optimizes only locally.
  - Each words(nodes) connect to their $K$ neighbors. The $K$ neighbors of a word is defined as the context of the word.
</details>

# Scalable RNN Language Model
<details>
<summary>
<a href="https://papers.nips.cc/paper/3583-a-scalable-hierarchical-distributed-language-model.pdf">A Scalable Hierarchical Distributed Language Model(2009)</a>
</summary>
</details>

<details>
<summary>
<a href="https://arxiv.org/abs/1602.02410">Exploring the Limits of Language Modeling</a>
</summary>
</details>
  
# Text Classification
<details>
<summary>
<a href="https://nlp.stanford.edu/pubs/SocherHuvalManningNg_EMNLP2012.pdf">Semantic Compositionality through Recursive Matrix-Vector(2012)</a>
</summary>

- Contributions
  - Matrix-vector representation of words + Recursive Neural Networks for semantic compositionality.
  - Showed to work well for:
    - learning compositionality for adverb-adjective pairs
    - learning boolean operators of propositional logic
    - sentiment detection
    - learning the semantic relationships between words
- How it works
  - Matrix-vector representation of words
    - Definition of compositionality
      - The ability to learn vector representations for various types of phrases and sentences of arbitrary length.
    - In terms of semantic compositionality, each words have two roles:
      - operand/constituent
        - a word(or a phrase) has a semantic meaning in itself.
        - to express this role as an operand, a vector is attached to a word.
      - operator
        - a word(or a phrase) will change the meaning of nearby words
        - to express this role as an operator, a matrix(linear transformation) is attached to a word.
  - Recursive Neural Networks
    - Split the inputs using a parse tree, and recursively composes the nodes using nonlinear transformations. i.e. applying recursive neural network on the nodes in a parse tree.
- Limitations
  - The performance is highly dependent on the parser.
  - Too many parameters to learn
  - Bias toward the topmost nodes in the parse tree
</details>

<details>
<summary>
<a href="http://www.aclweb.org/anthology/P14-1062">A Convolutional Neural Network for Modeling Sentences(2014)</a>
</summary>

- Contributions
  - Proposes DCNN, which automatically learns semantic/syntatic compositionality
  - DCNN automatically learns the feature graph, and does not rely on external parse tree.
- Related Model: Max-TDNN
  - Pros
    - Sensitive to word orders
    - Independent of external word features(e.g. parse tree)
  - Weaknesses
    - Narrow-type convolution: words at the margins are largely neglected
    - Max-pooling: the order of feature occurences is ignored- 
- How it works
  - A **Feature Map** is defined as the follows:
    - Wide-type convolution: mitigates the problem of words at the margin being neglected.
    - k-max pooling
      - (fixed)k-max pooling on the input layer
        - for a given sequence, accept only the top-k values as the output
        - the input to the intermediate layers will be independent of the length of the input sentences
      - Dynamic k-max pooling on the intermediate layers
        - k is a function of the input sentence length
    - Nonlinear transformation
  - Mutiple Feature Maps are stacked to form a DCNN model
- Properties
  - Sensitive to the word orders
  - **Convolution and pooling layers automatically build internal feature graphs over each inputs**
  - k-max pooling allows to draw features from words located relatively far from each other
</details>

<details>
<summary>
<a href="http://emnlp2014.org/papers/pdf/EMNLP2014181.pdf">Convolutional Neural Networks for Sentence Classification(2014)</a>
</summary>

- Contribution
  - Empirical evaluation of word-level CNNs for text classification tasks based on pretrained word embeddings(word2vec, Mikolov 2013).
- How it works
  - Comparison of CNN text classification models under various settings
    - Models
      - CNN + weight updates with random initialization
      - CNN + static pretrained word embeddings
      - CNN + pretrained word embeddings + fine tuning
      - CNN + fined-tuned word embeddings + pretrained word embeddings
  - CNNs + pretrained vectors + regularization(dropout, l2 regularization)
- Results
  - CNN text classifiers that leverage the pretrained vectors show state-of-the-art results in general, even without much tuning.
</details>

<details>
<summary>
<a href="https://arxiv.org/abs/1509.01626">Character-level Convolutional Neural Networks for Text Classifcation(2015)</a>
</summary>

- Contribution
  - First character-level CNN model for text classification
  - Comparable accuracy to traditional models, even without explicit features
- Limitations
  - Classification accuracy is somewhat questionable
</details>
  
# More Embeddings
<details>
<summary>
<a href="https://cs.stanford.edu/~quocle/paragraph_vector.pdf">Distributed Representation of Sentences and Documents(2014)</a>
</summary>

- Motivation
  - Traditional representations have drawbacks
    - BOW: word order is lost
    - Bag-Of-ngrams: word order is only preserved within short context, data sparsity and high dimensionality
- Contribution
  - An unsupervised model for learning representations for a sentence/paragraph that can predict words in a sentence/paragraph
- How it works
  - paragraph vectors are unique among paragraphs
  - word vectors are shared
  - paragraph vectors are learned so that it can predict the sampled words in the corresponding paragraphs well(just like word2vec)
- Limitation
</details>

<details>
<summary>
<a href="https://arxiv.org/pdf/1503.03578.pdf">LINE: Large-scale Information Network Embedding(2015)</a>
</summary>

- Contributions
  - Scalable Network Network Embedding algorithm that considers both 1st order and 2nd order proximity of a network.
- Proximity
  - 1st order proximity: Measures the direct connectedness between nodes
  - 2nd order proximity: Measures the shared neighborhood structures
- Applications
  - Word embedding: word networks constructed from a corpus
  - Social network
- Limitations
  - Loose integration of 1st & 2nd order proximities: The paper naively concatenates the network embeddings learnt seperately from optimizing each objectives.
  - Single-layer embedding
- Possible Improvements
  - Tighter integration of the learning objectives and parameters, using MRF formulation together with Variational Inference, maybe?
  - Multi-layer embedding
</details>

# Deep Learning & Generative Models
<details>
<summary>
<a href="https://papers.nips.cc/paper/4613-a-neural-autoregressive-topic-model.pdf">A neural autoregressive topic model(2012)</a>
</summary>

- Motivations
  - Replicated Softmax, a generalization of RBMs to model topics, was too slow to train on documents.
- Contributions
  - Low-computational complexity generative model for topic modelling
- How it works
  - Modifies the structure of the Replicated Softmax to lower the computational complexity
  - Feedforward structure, where each conditional probability is computed by a tree of binary logistic regressions.
  - Borrows some structure from NADE to obtain an efficient way to share the hidden layer parameters across the conditionals.
  
</details>

<details>
<summary>
<a href="https://arxiv.org/pdf/1511.06038.pdf">Neural Variational Inference for Text Processing(2015)</a>
</summary>

- Motivation
  - Traditional generative models were either too computationally heavy(MCMC), or too biased(Variational Inference)
- Contributions
  - VAE approach to Topic Modeling
- Why it works well
  - Latent variables give the ability to sum over all the possibilites in terms of semantics. i.e. Latent variables mitigates the overfitting.
</details>

# Sentiment Analysis
<details>
<summary>
<a href="https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf">Hierarchical Attention networks for Document Classification(2016 NAACL)</a>
</summary>

- Contributions
  - Bottom-up compositional model for modeling document hierarchy
  - Exploits the attention mechanism to select important words & sentences that are importance for sentence & document representations
</details>

<details>
<summary>
<a href="https://aclweb.org/anthology/D16-1058">Attention-based LSTM for Aspect-level Sentiment Classification(2016 EMNLP)</a>
</summary>

- Contribution
  - First paper to use aspect embedding for sentiment classification task
  - Attention + LSTM + aspect embedding

- Different versions
  - version 1: Concatenate aspect embedding and hidden state vectors of LSTMs
  - version 2: Concatenate aspect embedding to the hidden states AND the inputs
</details>

# Machine Translation
<details>
<summary>
<a href="http://www.aclweb.org/anthology/J93-2003">The Mathematics of Statistical Machine Translation: Parameter Estimation(2003)</a>
</summary>
</details>

<details>
<summary>
<a href="https://arxiv.org/pdf/1609.07730.pdf">Lattice-Based Recurrent Neural Network Encoders for Neural Machine Translation(2016)</a>
</summary>
</details>

<details>
<summary>
<a href="https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf">Sequence to Sequence Learning with Neural Networks(2014)</a>
</summary>
</details>

<details>
<summary>
<a href="https://arxiv.org/abs/1409.0473">Neural machine translation by jointly learning to align and translate</a>
</summary>
</details>

# Parsing
<details>
<summary>
<a href="https://cs.stanford.edu/~danqi/papers/emnlp2014.pdf">A Fast and Accurate Dependency Parser using Neural Networks(2014)</a>
</summary>
</details>

<details>
<summary>
<a href="http://www.cs.cmu.edu/~lingwang/papers/acl2015.pdf">Transition-Based Dependency Parsing with Stack Long Short-Term Memory(2015)</a>
</summary>
</details>

# Question Answering
<details>
<summary>
<a href="http://papers.nips.cc/paper/5945-teaching-machines-to-read-and-comprehend.pdf">Teaching Machines to Read and Comprehend(2015)</a>
</summary>
</details>

<details>
<summary>
<a href="https://arxiv.org/pdf/1506.02075.pdf">Large-scale Simple Question Answering with Memory Networks(2015)</a>
</summary>
</details>

<details>
<summary>
<a href="https://arxiv.org/pdf/1506.07285.pdf">Dynamic Memory Networks for Natural Language Processing(2015)</a>
</summary>
</details>

<details>
<summary>
<a href="https://arxiv.org/pdf/1502.05698.pdf">Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks(2015)</a>
</summary>
</details>

# Conversation Modeling
<details>
<summary>
<a href="https://arxiv.org/pdf/1510.03055.pdf">A Diversity-Promoting Objective Function for Neural Conversation Models(2015)</a>
</summary>
</details>

<details>
<summary>
<a href="https://arxiv.org/pdf/1507.04808.pdf">Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models(2015)</a>
</summary>
</details>

# Image Captioning
<details>
<summary>
<a href="https://arxiv.org/pdf/1411.4555.pdf">Show and Tell: A Neural Image Caption Generator(2014)</a>
</summary>
</details>

<details>
<summary>
<a href="https://arxiv.org/pdf/1411.5654.pdf">Learning a recurrent visual representation for image caption generation(2014)</a>
</summary>
</details>

<details>
<summary>
<a href="https://cs.stanford.edu/people/karpathy/cvpr2015.pdf">Deep Visual-Semantic Alignments for Generating Image Descriptions(2015)</a>
</summary>
</details>

<details>
<summary>
<a href="https://arxiv.org/pdf/1502.03044.pdf">Show, attend and tell: Neural image caption generation with visual attention(2015)</a>
</summary>
</details>

# Summarization
<details>
<summary>
<a href="http://aclweb.org/anthology/D17-1222">Extractive Summarization Using Multi-Task Learning with Document Classification(2017 EMNLP)</a>
</summary>
</details>

<details>
<summary>
<a href="http://www.aclweb.org/anthology/P16-1046">Neural Summarization by Extracting Sentences and Words(2016 ACL)</a>
</summary>
</details>