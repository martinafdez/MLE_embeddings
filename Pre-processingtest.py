import os, re
import pickle

import gensim
from gensim.corpora.dictionary import Dictionary
from gensim import corpora, similarities, models
from gensim.models import Word2Vec
from gensim.parsing import strip_punctuation, stem_text

import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


# input
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()


#access corpus
#sentences = MySentences("/Users/martinafernandez/Desktop/CorpusStandard/")


#regular expression pre-processing
repdict = {'G' : 'gangster', 'P' : 'paper'}
def regepxreprocess (sentences, repdict):
    for sentence in sentences:
        pattern = re.compile("r'^[g|G]/b|$', r'^[p|P]/b|$'".join([re.escape(k) for k in repdict.keys()]), re.M)
        return pattern.sub(lambda x: repdict[x.group(0)], sentences)


# grouping related words to one corresponding value
dic = {('beefing', 'beefin') : 'beef', ('blud', 'blad') : 'blood', ('bro', 'bruv', 'brudda', 'bredda') : 'brother', 'bredrin' : 'brethren', 'cuz' : 'cousin', ('dipping', 'dippin') : 'dip', 'fam' : 'family', ('federal agent', 'federal officer', 'fed') : 'feds', ('gangsta', 'gansta') : 'gangster', 'paigon' : 'pagan', 'penitentiary' : 'pen', ('spittin', 'spitting') : 'spit'}
#def replace_all(sentences, dic):
def replace_all(w, dic):
#    sentence = []
#    for sentence in sentences:
#     words = sentences.split()
#     for w in words:
     for k, v in dic.iteritems():
         if w in k:
             return v
#             sentences = sentences.replace(k, v)
#             return sentences
 #            sentence.append(sentences)
 #            return sentence

#lemmatisation pre-processing
def lemmatizingpreprocess(w):
#    sentence = sentences.split()
#    for sentence in sentences:
#     words = sentences.split()
#     words = ['gassed', 'dipped', 'dipping', 'spat', 'beefing', 'pagans']
 #    for word in words:
     lemma = WordNetLemmatizer().lemmatize(w)
     return lemma

# define corpus dictionary
corpus={}
dirpath = '/Users/martinafernandez/Desktop/CorpusStandard/'
#dirpath = "/Users/martinafernandez/Desktop/CorpusStandard/"
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
            for sent in rawsents:
                if not re.search('^\s*\r\n$', sent):  # ignore empty lines
                    #sent = sent.lower()  stem_text() does this
                    sent = strip_punctuation(sent)  # run these gensim functions first on string; returns string
                    sent = stem_text(sent)  # n.b. lower cases too
                    sent = regepxreprocess(sent, repdict)
                    wordtokens = word_tokenize(sent)  # run this last as it returns a list
                    ## final step: input to gensim should be a list of sentences (where each sentence is a list of words)
                    processed = []  # empty list for this sentence
                    for word in wordtokens:
                        word = replace_all(word, dic)  # replace first
                        word = lemmatizingpreprocess(word)  # then lemmatise, is that right?
                        processed.append(word)
                    # now check if you've already started the word list for this subcorpus
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
words = list(model.wv.vocab)
print(words)
# access vector for one word
print(model['air'])
# save model
model.save('BNCPracticeEmbeddings-airP.bin')
# load model
new_model = Word2Vec.load('BNCPracticeEmbeddings-airP.bin')
print(new_model)
