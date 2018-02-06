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
# targetword = sys.argv[1]
targetword = 'air'

# regular expression pre-processing
repdict = {'G': 'gangster', 'P': 'paper'}


def regepxreprocess(sentences, repdict):
    for sentence in sentences:
        pattern = re.compile("r'^[g|G]/b|$', r'^[p|P]/b|$'".join([re.escape(k) for k in repdict.keys()]), re.M)
        return pattern.sub(lambda x: repdict[x.group(0)], sentences)


# grouping related words to one corresponding value
dic = {('beefing', 'beefin'): u'beef', ('blud', 'blad'): u'blood', ('bro', 'bruv', 'brudda', 'bredda'): u'brother',
       'bredrin': u'brethren', 'cuz': u'cousin', ('dipping', 'dippin'): u'dip', 'fam': u'family',
       ('federal agent', 'federal officer', 'fed'): u'feds', ('gangsta', 'gansta'): u'gangster', 'paigon': u'pagan',
       'penitentiary': u'pen', ('spittin', 'spitting'): u'spit'}


def replace_all(w, dic):
    for k, v in dic.iteritems():
        for tok in k:
            if w == tok:
                w = v
    return w


# lemmatisation pre-processing
def lemmatizingpreprocess(w):
    lemma = WordNetLemmatizer().lemmatize(w)
    return lemma


# define corpus dictionary
corpus = {}
#dirpath = '/Users/apc38/Dropbox/workspace/gitHub/wordtracker/MLEembeddings_cainesap/Corpus'  # Andrew
dirpath = '/Users/martinafernandez/Desktop/Corpus/'  # Martina
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
                print(name)
                print('sentence %d' % snum)
                print(sent)
                sent = strip_punctuation(sent)  # rm punctuation from sentence string; returns string
                if sent is not None and re.search('\w', sent):  # ignore empty lines
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
# model = gensim.models.Word2Vec(dirpath, size = 300, min_count= 1)
## 2 choices here: you could have a dictionary of models (I think), one for each subcorpus;
# i.e. for subcorp, sentences in corpus.iteritems():
## or build them separately line by line
# e.g.
#modelStandard80s = gensim.models.Word2Vec(corpus['Standard1980s'], size=300, min_count=1)
#modelMLE10s = gensim.models.Word2Vec(corpus['MLE2010s'], size=300, min_count=1)
# model = gensim.models.Word2Vec(dirpath, size = 300, min_count= 1)
# print(model)

for corps, sentences in corpus.iteritems():
    model = gensim.models.Word2Vec(corpus)
    words = list(model.wv.vocab)
    with open ('model + vocab', 'wb'):
        pickle.dump(model, words)




# summarize vocabulary
#words_stand = list(modelStandard80s.wv.vocab)
# print(words_stand)
#words_MLE = list(modelMLE10s.wv.vocab)
# print(words_MLE)
# access vector for one word
#print(modelStandard80s['air'])
#print(modelMLE10s['air'])
# save model
#modelStandard80s.save('BNCPracticeEmbeddings-air0.bin')
#modelMLE10s.save('MLEPracticeEmbeddings-air0.bin')
# load model
#new_modelstand = Word2Vec.load('BNCPracticeEmbeddings-air0.bin')
#new_modelMLE = Word2Vec.load('MLEPracticeEmbeddings-air0.bin')
#print(new_modelstand)
#print(new_modelMLE)

