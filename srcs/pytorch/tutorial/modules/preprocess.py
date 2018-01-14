from nltk import sent_tokenize, wordpunct_tokenize
from collections import Counter
from itertools import chain
import torch
from torch.autograd import Variable
import numpy as np

class Vocab:
    '''Abstract vocabulary class that has useful helper functions
    '''
    def __init__(self, corpus, lower = False, sos='<sos>', eos='<eos>', unk = '<unk>', pad = '<pad>', **kwargs):
        '''
        Builds a vocabulary, using the words in the corpus

        Args:
            corpus: string of text.
            max_size: the maximum number of words to be included in the vocabulary, excluding the special tokens(sos, eos, unk, pad)
            min_freq: the minumum number of times a word should appear to be included in the vocabulary.
            emb_dim: the dimension of the word embeddings. If not specified, equal to the number of words in the vocabulary.
            (to support one-hot encoding)
            one_hot: whether to use one_hot encodings

        Returns:
            word2id: dictionary of (word, id) pairs
            id2word: list of words in the vocabulary
        '''
        for k, v in kwargs.items():
            setattr(self, k, v)
        if not hasattr(self, 'max_size'): self.max_size = 100
        if not hasattr(self, 'min_freq'): self.min_freq = 1
        
        self.lower = lower
        self.sos = sos
        self.eos = eos
        self.unk = unk
        self.pad = pad
        
        if lower: corpus = corpus.lower()
        freq = Counter(wordpunct_tokenize(corpus)).most_common(self.max_size)
        freq = {w:c for (w,c) in freq if c >= self.min_freq}

        id2word =  sorted(freq.keys()) + [pad, unk, sos, eos]
        word2id = {word: i for i, word in enumerate(id2word)}
        
        num_embedding = len(id2word)
        if not hasattr(self, 'embedding_dim'): embedding_dim = num_embedding
        emb = torch.nn.Embedding(num_embedding, embedding_dim)
        if hasattr(self, 'one_hot'):
            if self.one_hot and embedding_dim == num_embedding:
                for i in range(emb.weight.data.size(0)):
                    emb.weight.data[i,:] = 0
                    emb.weight.data[i,i] = 1
            else: raise Exception('embedding_dim should match the num_embeddings for one-hot encoding')

        self.freq = freq
        self.id2word = id2word
        self.word2id = word2id
        self.emb = emb
             
    def __len__(self):
        return len(self.id2word)
    
    def __str__(self):
        return str(self.word2id)
    
    def __getitem__(self, key):
        if type(key) == int:
            return self.id2word[key]
        elif type(key) == str:
            try:
                return self.word2id[key]
            except KeyError:
                return self.word2id[self.unk]
        else:
            raise KeyError('Key type should be either int or string')

    def text2id(self, text):
        '''Directly map a text into a list of indices
        
        Args:
            text: string
        '''
        
        return torch.LongTensor([self[w] for w in wordpunct_tokenize(text)])
    
    def sents2id(self, text):
        '''Tokenizes a text into a list of sentences, mapping the words to corresponding indices.

        Args:
            text: string

        Returns:
            sents_list: List of torch.LongTensor, where each elements are words indices in the sentences
        '''
        if self.lower: text = text.lower()    
        sents = sent_tokenize(text)

        sos_tensor = torch.LongTensor([self[self.sos]])
        eos_tensor = torch.LongTensor([self[self.eos]])
        sents_list = []
        for i in range(len(sents)):
            sent = torch.cat([sos_tensor, self.text2id(sents[i]), eos_tensor], 0)
            sents_list.append(sent)

        return sents_list
    
    def id2text(self, indices):
        return ' '.join([self[i_word] for i_word in indices])

    def id2sents(self, sents):
        '''Returns the string representation of the sentences, where sentences is a list of sentences
        and each sentences are lists of word ids.

        Args:
            sents: a torch.LongTensor of word ids in the dictionary

        Returns:
            sents_str: string representation of sentences.
        '''

        return ' '.join([self[i_word] for i_word in chain(*sents)])
    
    def id2emb(self, indices):
        '''
        Converts an index into the corresponding embedding
        
        Args:
            indices: torch.LongTensor
        '''
        return self.emb(Variable(indices)).data
    
    def text2emb(self, text):
        '''
        Tokenizes a text into the corresponding embeddings

        Args:
            text: string

        Returns:
            a sequence of embeddings corresponding to tokens.
        '''
    
        return self.id2emb(self.text2id(text))