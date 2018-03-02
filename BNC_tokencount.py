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
import collections
from collections import Counter


## which word?
# targetword = sys.argv[1]
#targetword = 'air'

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
                sent = strip_punctuation(sent)  # rm punctuation from sentence string; returns string
                if sent is not None and re.search('\w', sent):  # ignore empty lines
                    if not re.search('^\s*\r\n$', sent):
                        #sent = stem_text(sent)  # gensim stemmer: n.b. lower cases too
                        sent = regepxreprocess(sent, repdict)
                        wordtokens = sent.lower().split()
                        ## final step: input to gensim should be a list of sentences (where each sentence is a list of words)
                        processed = []  # empty list for this sentence
                        for word in wordtokens:
                            word = replace_all(word, dic)  # replace first
                            #word = lemmatizingpreprocess(word)  # then lemmatise
                            processed.append(word)
                        ## now check if you've already started the word list for this subcorpus
                        if subcorp in corpus:
                            corpus[subcorp].append(processed)  # if yes, add to list of sentences
                        else:
                            corpus[subcorp] = [processed]  # else, start a list

# words of interest
words = {'air', 'gas', 'gassed', 'beef', 'butters', 'spice', 'spicy', 'rinse', 'rinsed', 'rinsing', 'jam', 'dip', 'dough', 'bread', 'food', 'sauce', 'whip', 'whipping', 'pepper', 'dry', 'dead', 'hot', 'cold', 'hard', 'soft', 'myth', 'pagan', 'ghost', 'spit', 'lick', 'licked', 'body', 'buff', 'clapped', 'clap', 'man', 'blood', 'safe', 'deep', 'peak', 'long', 'ends', 'bait', 'paper', 'bars', 'spray', 'bare', 'feds', 'pen', 'gangster', 'dutty', 'yard', 'dem', 'nuff', 'ting', 'bun', 'brother', 'cousin', 'brethren', 'family', 'don', 'cheddar', 'moist', 'crumb', 'coin', 'corn', 'roll', 'bounce', 'shook'}

for corps, sentences in corpus.iteritems():
    # start word counter for each
    wordcount = Counter()
    for word in words:
        wordcount[word] = 0
    for sentence in sentences:
        tokens = sentence.split()
        for token in tokens:
            if token in words:
                wordcount[token] += 1
    for word in words:
        count = wordcount[word]
        print('%s, %s, %i' % (corpus, word, count))
