# Word embeddings
## NLP (Almost) from Scratch(2011)
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

## Distributed Representations of Words and Phrases and their Compositionality(2013)
- Contribution
  - Efficient learning algorthm for word embedding tasks, using
    - Skip-gram
    - Negative Sampling
    - Subsampling
  - Good performance in semantic & syntatic analogy tasks

- How it works
  - Negative Sampling: Learn word embeddings so that the embeddings can be used to discriminate the words from the training corpus against the noise words.
  - Subsampling: during training, ignore frequent words by a certain probability(somewhat similar to dropouts)

- Limitations
  - One word, one embedding -> polysemy, or global word context is ignored
  - Cannot utilize the morphological information well: car != cars