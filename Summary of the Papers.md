# Basic Word embeddings
### [NLP (Almost) from Scratch(2011)](http://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf)
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

### [Distributed Representations of Words and Phrases and their Compositionality(2013)](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
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

# Multilingual word embedding
### Mutilingual word embedding in general
- Motivation for Multilingual word embedding
  - Goal: Learns word embeddings that can generalize over *different languages* and *different  NLP tasks*
  - Why it is needed: For Transfer Learning. Labeled text data is abundant for only few languages(e.g. English), so it may be a good idea to transfer the representations from the languages with abundant labeled text data to the languages with scarce text data for a NLP task.
### [Bilbowa: Fast bilingual distributed representations without word alignments(2014)](https://arxiv.org/abs/1410.2455)
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
### [An Autoencoder Approach to Learning Bilingual Word Representations(2014)](https://papers.nips.cc/paper/5270-an-autoencoder-approach-to-learning-bilingual-word-representations.pdf)
- How it works
  - Uses Autoencoder to obtain latent representation of bilingual representation of words
  - A bag-of-words in language $A$ will be reconstructed to the corresponding bag-of-words in another language $B$
- vs Bilbowa
  - Nonlinear transformation of inputs
- Limitations
  - Evaluated on the limited dataset.(EN-German)

# Language Modeling
### [An Empirical Study of Smoothing Techniques for Language Modeling(1996)](http://aclweb.org/anthology/P96-1041)
### [A Neural Probabilistic Language Model(2003)](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

# Contextual Language Model
### [Document Context Language Models(2015)](https://arxiv.org/abs/1511.03962)
### [A Fast Variational Approach for Learning Markov Random Field Language Models(2015)](http://proceedings.mlr.press/v37/jernite15.pdf)

# Scalable RNN Language Model
### [A Scalable Hierarchical Distributed Language Model(2009)](https://papers.nips.cc/paper/3583-a-scalable-hierarchical-distributed-language-model.pdf)
### [Exploring the Limits of Language Modeling](https://arxiv.org/abs/1602.02410)
  
# Text Classification
### [Semantic Compositionality through Recursive Matrix-Vector(2012)](https://nlp.stanford.edu/pubs/SocherHuvalManningNg_EMNLP2012.pdf)
### [A Convolutional Neural Network for Modeling Sentences(2014)](http://www.aclweb.org/anthology/P14-1062)
### [Convolutional Neural Networks for Sentence Classification(2014)](http://emnlp2014.org/papers/pdf/EMNLP2014181.pdf)
### [Character-level Convolutional Neural Networks for Text Classifcation(2015)](https://arxiv.org/abs/1509.01626)
  
# More Embeddings
### [Distributed Representation of Sentences and Documents(2014)](https://cs.stanford.edu/~quocle/paragraph_vector.pdf)
### [LINE: Large-scale Information Network Embedding(2015)](https://arxiv.org/pdf/1503.03578.pdf)

# Deep Learning & Generative Models
### [A neural autoregressive topic model(2012)](https://papers.nips.cc/paper/4613-a-neural-autoregressive-topic-model.pdf)
### [Neural Variational Inference for Text Processing(2015)](https://arxiv.org/pdf/1511.06038.pdf)

# Sentiment Analysis
### [Recursive Deep Models for Semantic Compositionality over a Sentiment Treebank(2013)](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf)
### [Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks(2015)](http://www.aclweb.org/anthology/P15-1150)

# Machine Translation
### [The Mathematics of Statistical Machine Translation: Parameter Estimation(2003)](http://www.aclweb.org/anthology/J93-2003)
### [Lattice-Based Recurrent Neural Network Encoders for Neural Machine Translation(2016)](https://arxiv.org/pdf/1609.07730.pdf)
### [Sequence to Sequence Learning with Neural Networks(2014)](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
### [Neural machine translation by jointly learning to align and translate](https://arxiv.org/abs/1409.0473)

# Parsing
### [Learning to Map Sentences to Logical Form: Structured Classification with Probabilistic Categorial Grammars(2005)](https://homes.cs.washington.edu/~lsz/papers/zc-uai05.pdf)
### [Parsing with Compositional Vector Grammers(2013)](https://nlp.stanford.edu/pubs/SocherBauerManningNg_ACL2013.pdf)
### [A Fast and Accurate Dependency Parser using Neural Networks(2014)](https://cs.stanford.edu/~danqi/papers/emnlp2014.pdf)
### [Transition-Based Dependency Parsing with Stack Long Short-Term Memory(2015)](http://www.cs.cmu.edu/~lingwang/papers/acl2015.pdf)

# Question Answering
### [Teaching Machines to Read and Comprehend(2015)](http://papers.nips.cc/paper/5945-teaching-machines-to-read-and-comprehend.pdf)
### [Large-scale Simple Question Answering with Memory Networks(2015)](https://arxiv.org/pdf/1506.02075.pdf)
### [Dynamic Memory Networks for Natural Language Processin(2015)](https://arxiv.org/pdf/1506.07285.pdf)
### [Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks(2015)](https://arxiv.org/pdf/1502.05698.pdf)

# Conversation Modeling
### [A Diversity-Promoting Objective Function for Neural Conversation Models(2015)](https://arxiv.org/pdf/1510.03055.pdf)
### [Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models(2015)](https://arxiv.org/pdf/1507.04808.pdf)

# Image Captioning
### [Show and Tell: A Neural Image Caption Generator(2014)](https://arxiv.org/pdf/1411.4555.pdf)
### [Learning a recurrent visual representation for image caption generation(2014)](https://arxiv.org/pdf/1411.5654.pdf)
### [Deep Visual-Semantic Alignments for Generating Image Descriptions(2015)](https://cs.stanford.edu/people/karpathy/cvpr2015.pdf)
### [Show, attend and tell: Neural image caption generation with visual attention(2015)](https://arxiv.org/pdf/1502.03044.pdf)