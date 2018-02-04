import os, re
import pickle
import numpy as np

import gensim
from gensim.corpora.dictionary import Dictionary
from gensim import corpora, similarities, models
from gensim.models import Word2Vec
from gensim.parsing import strip_punctuation, stem_text

import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

from scipy import spatial


## which word?
#targetword = sys.argv[1]
targetword = 'air'


#regular expression pre-processing
repdict = {'G' : 'gangster', 'P' : 'paper'}
def regepxreprocess (sentences, repdict):
    for sentence in sentences:
        pattern = re.compile("r'^[g|G]/b|$', r'^[p|P]/b|$'".join([re.escape(k) for k in repdict.keys()]), re.M)
        return pattern.sub(lambda x: repdict[x.group(0)], sentences)


# grouping related words to one corresponding value
dic = {('beefing', 'beefin') : u'beef', ('blud', 'blad') : u'blood', ('bro', 'bruv', 'brudda', 'bredda') : u'brother', 'bredrin' : u'brethren', 'cuz' : u'cousin', ('dipping', 'dippin') : u'dip', 'fam' : u'family', ('federal agent', 'federal officer', 'fed') : u'feds', ('gangsta', 'gansta') : u'gangster', 'paigon' : u'pagan', 'penitentiary' : u'pen', ('spittin', 'spitting') : u'spit'}
def replace_all(w, dic):
    for k, v in dic.iteritems():
        for tok in k:
            if w==tok:
                w = v
    return w

#lemmatisation pre-processing
def lemmatizingpreprocess(w):
     lemma = WordNetLemmatizer().lemmatize(w)
     return lemma

# define corpus dictionary
corpus={}
dirpath = '/Users/apc38/Dropbox/workspace/gitHub/wordtracker/MLEembeddings_cainesap/Corpus'  # Andrew
#dirpath = '/Users/martinafernandez/Desktop/CorpusStandard/'  # Martina
for root, dirs, files in os.walk(dirpath):
    # all files in directory
    for name in files:
        filepath = os.path.join(root, name)  # construct filepath
        if re.search('.txt$', filepath):  # check it's a text file
            subcorp = re.search('\w+\d{4}s', filepath).group()  # get decade and text type
            with open(filepath) as f:
                rawsents = f.readlines()  # load text as a list of sentences
            f.close()
            ## sentence clean up
            for snum, sent in enumerate(rawsents):
                print('sentence %d' % snum)
                sent = strip_punctuation(sent)  # rm punctuation from sentence string; returns string
                if sent is not None:  # ignore empty lines
                    if not re.search('^\s*\r\n$', sent):
                        sent = stem_text(sent)  # gensim stemmer: n.b. lower cases too
                        sent = regepxreprocess(sent, repdict)
                        wordtokens = word_tokenize(sent)  # run this last as it returns a list
                        ## final step: input to gensim should be a list of sentences (where each sentence is a list of words)
                        processed = []  # empty list for this sentence
                        for word in wordtokens:
                            word = replace_all(word, dic)  # replace first
                            word = lemmatizingpreprocess(word)  # then lemmatise, is that right?
                            processed.append(word)
                        ## now check if you've already started the word list for this subcorpus
                        if subcorp in corpus:
                            corpus[subcorp].append(processed)  # if yes, add to list of sentences
                        else:
                            corpus[subcorp] = [processed]  # else, start a list


# train models
#model = gensim.models.Word2Vec(dirpath, size = 300, min_count= 1)
## 2 choices here: you could have a dictionary of models (I think), one for each subcorpus;
# i.e. for subcorp, sentences in corpus.iteritems():
## or build them separately line by line
# e.g.
modelStandard80s = gensim.models.Word2Vec(corpus['Standard1980s'], size = 300, min_count= 1)
modelMLE10s = gensim.models.Word2Vec(corpus['MLE2010s'], size = 300, min_count= 1)
#model = gensim.models.Word2Vec(dirpath, size = 300, min_count= 1)
#print(model)

# summarize vocabulary
words_stand = list(modelStandard80s.wv.vocab)
#print(words_stand)
words_MLE = list(modelMLE10s.wv.vocab)
#print(words_MLE)
# access vector for one word
print(modelStandard80s['air'])
print(modelMLE10s['air'])
# save model
modelStandard80s.save('BNCPracticeEmbeddings-air0.bin')
modelMLE10s.save('MLEPracticeEmbeddings-air0.bin')
# load model
new_modelstand = Word2Vec.load('BNCPracticeEmbeddings-air0.bin')
new_modelMLE = Word2Vec.load('MLEPracticeEmbeddings-air0.bin')
print(new_modelstand)
print(new_modelMLE)


## 2 functions below from Ryan Heuser https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf
def intersection_align_gensim(m1, m2, words=None):
    """
    Intersect two gensim word2vec models, m1 and m2.
    Only the shared vocabulary between them is kept.
    If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
    Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
    These indices correspond to the new syn0 and syn0norm objects in both gensim models:
        -- so that Row 0 of m1.syn0 will be for the same word as Row 0 of m2.syn0
        -- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
    The .vocab dictionary is also updated for each model, preserving the count but updating the index.
    """
    
    # Get the vocab for each model
    vocab_m1 = set(m1.vocab.keys())
    vocab_m2 = set(m2.vocab.keys())
    
    # Find the common vocabulary
    common_vocab = vocab_m1 & vocab_m2
    if words: common_vocab &= set(words)
    
    # If no alignment necessary because vocab is identical...
    if not vocab_m1 - common_vocab and not vocab_m2 - common_vocab:
        return (m1, m2)
    
    # Otherwise sort by frequency (summed for both)
    common_vocab = list(common_vocab)
    common_vocab.sort(key=lambda w: m1.vocab[w].count + m2.vocab[w].count, reverse=True)
    
    # Then for each model...
    for m in [m1, m2]:
        # Replace old syn0norm array with new one (with common vocab)
        indices = [m.vocab[w].index for w in common_vocab]
        old_arr = m.syn0norm
        new_arr = np.array([old_arr[index] for index in indices])
        m.syn0norm = m.syn0 = new_arr
        
        # Replace old vocab dictionary with new one (with common vocab)
        # and old index2word with new one
        m.index2word = common_vocab
        old_vocab = m.vocab
        new_vocab = {}
        for new_index, word in enumerate(common_vocab):
            old_vocab_obj = old_vocab[word]
            new_vocab[word] = gensim.models.word2vec.Vocab(index=new_index, count=old_vocab_obj.count)
        m.vocab = new_vocab
    
    return (m1, m2)

def smart_procrustes_align_gensim(base_embed, other_embed, words=None):
    """Procrustes align two gensim word2vec models (to allow for comparison between same word across models).
    Code ported from HistWords <https://github.com/williamleif/histwords> by William Hamilton <wleif@stanford.edu>.
        (With help from William. Thank you!)
    First, intersect the vocabularies (see `intersection_align_gensim` documentation).
    Then do the alignment on the other_embed model.
    Replace the other_embed model's syn0 and syn0norm numpy matrices with the aligned version.
    Return other_embed.
    If `words` is set, intersect the two models' vocabulary with the vocabulary in words (see `intersection_align_gensim` documentation).
    """
    
    # make sure vocabulary and indices are aligned
    in_base_embed, in_other_embed = intersection_align_gensim(base_embed, other_embed, words=words)
    
    # get the embedding matrices
    base_vecs = in_base_embed.syn0norm
    other_vecs = in_other_embed.syn0norm
    
    # just a matrix dot product with numpy
    m = other_vecs.T.dot(base_vecs)
    # SVD method from numpy
    u, _, v = np.linalg.svd(m)
    # another matrix operation
    ortho = u.dot(v)
    # Replace original array with modified one
    # i.e. multiplying the embedding matrix (syn0norm)by "ortho"
    other_embed.syn0norm = other_embed.syn0 = (other_embed.syn0norm).dot(ortho)
    return other_embed

MLEaligned = smart_procrustes_align_gensim(new_modelstand, new_modelMLE)
## which words?
u = new_modelstand[targetword]
print(u)
v = MLEaligned[targetword]
#scipy.spatial.distance.cosine(u, v)
