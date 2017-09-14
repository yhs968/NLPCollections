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
  - Scales linearly
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
</details>

# Contextual Language Model
<details>
<summary>
<a href="https://arxiv.org/abs/1511.03962">Document Context Language Models(2015)</a>
</summary>
</details>

<details>
<summary>
<a href="http://proceedings.mlr.press/v37/jernite15.pdf">A Fast Variational Approach for Learning Markov Random Field Language Models(2015)</a>
</summary>
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
</details>

<details>
<summary>
<a href="http://www.aclweb.org/anthology/P14-1062">A Convolutional Neural Network for Modeling Sentences(2014)</a>
</summary>
</details>

<details>
<summary>
<a href="http://emnlp2014.org/papers/pdf/EMNLP2014181.pdf">Convolutional Neural Networks for Sentence Classification(2014)</a>
</summary>
</details>

<details>
<summary>
<a href="https://arxiv.org/abs/1509.01626">Character-level Convolutional Neural Networks for Text Classifcation(2015)</a>
</summary>
</details>
  
# More Embeddings
<details>
<summary>
<a href="https://cs.stanford.edu/~quocle/paragraph_vector.pdf">Distributed Representation of Sentences and Documents(2014)</a>
</summary>
</details>

<details>
<summary>
<a href="https://arxiv.org/pdf/1503.03578.pdf">LINE: Large-scale Information Network Embedding(2015)</a>
</summary>
</details>

# Deep Learning & Generative Models
<details>
<summary>
<a href="https://papers.nips.cc/paper/4613-a-neural-autoregressive-topic-model.pdf">A neural autoregressive topic model(2012)</a>
</summary>
</details>

<details>
<summary>
<a href="https://arxiv.org/pdf/1511.06038.pdf">Neural Variational Inference for Text Processing(2015)</a>
</summary>
</details>

# Sentiment Analysis
<details>
<summary>
<a href="https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf">Recursive Deep Models for Semantic Compositionality over a Sentiment Treebank(2013)</a>
</summary>
</details>

<details>
<summary>
<a href="http://www.aclweb.org/anthology/P15-1150">Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks(2015)</a>
</summary>
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
<a href="https://homes.cs.washington.edu/~lsz/papers/zc-uai05.pdf">Learning to Map Sentences to Logical Form: Structured Classification with Probabilistic Categorial Grammars(2005)</a>
</summary>
</details>

<details>
<summary>
<a href="https://nlp.stanford.edu/pubs/SocherBauerManningNg_ACL2013.pdf">Parsing with Compositional Vector Grammers(2013)</a>
</summary>
</details>

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