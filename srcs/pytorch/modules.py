from nltk import sent_tokenize, wordpunct_tokenize
from collections import Counter
from itertools import chain

def build_vocab(corpus, top_k = None):
    '''
    Builds a vocabulary, using the words in the corpus
    
    Args:
        corpus: string of text.
        top_k: the number of words to be included in the vocabulary, including the special tokens:
        "UNK(unknown)", "_BEGIN_(beginning of a sentence)", "_END_(end of a sentence)"
        
    Returns:
        word2id: dictionary of (word, id) pairs
        id2word: list of words in the vocabulary
    '''
    if type(top_k) == int:
        top_k -= 3
    word_counts = Counter(wordpunct_tokenize(corpus)).most_common(top_k)
    
    id2word = sorted([word for word,count in word_counts]) + ['UNK','_BEGIN_','_END_']
    word2id = {word: i for i, word in enumerate(id2word)}
    
    return word2id, id2word

def sents2id(corpus, top_k = None, case_sensitive = False):
    '''Tokenizes the whole corpus into sentences, mapping the words to corresponding indices.
    
    Args:
        corpus: string.
        
    Returns:
        sents_list: List of sentences, where each sentences are the list of word indices.
        word2id: dictionary of (word, id) pairs
        id2word: list of words in the vocabulary
        
    '''
    if not case_sensitive:
        corpus = corpus.lower()
    word2id, id2word = build_vocab(corpus, top_k)
    sents = sent_tokenize(corpus)
    
    sents_list = []
    for i in range(len(sents)):
        sent = wordpunct_tokenize(sents[i])
        sent = [word2id[word] if word in word2id else word2id['UNK'] for word in sent]
        sent = [word2id['_BEGIN_']] + sent + [word2id['_END_']]
        sents_list.append(sent)
        
    return sents_list, word2id, id2word

def id2sents(sents):
    '''Returns the string representation of the sentences, where sentences is a list of sentences
    and each sentences are lists of word ids.
    
    Args:
        sents: a list of word ids in the dictionary
        
    Returns:
        sents_str: string representation of sentences.
    '''
    
    return ' '.join([id2word[i_word] for i_word in chain(*sents)])