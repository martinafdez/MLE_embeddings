import gensim
import numpy as np
import pickle

from gensim.models import KeyedVectors
import scipy
from scipy import spatial


#loading models and accessing vectors
targetword = 'air'
corpus1 = 'Standard2010s'
corpus2 = 'MLE2010s'

#with open ('Standard1970s.vocab', 'rb') as f:
#    S1970 = pickle.load(f)

#with open ('Standard1980s.vocab', 'rb') as f:
#    S1980 = pickle.load(f)

#with open ('Standard1990s.vocab', 'rb') as f:
#    S1990 = pickle.load(f)

#with open ('Standard2000s.vocab', 'rb') as f:
#    S2000 = pickle.load(f)

#with open ('Standard2010s.vocab', 'rb') as f:
#    S2010vocab = pickle.load(f)

#with open ('MLE2010S.vocab', 'rb') as f:
#    MLE2010vocab = pickle.load(f)

#print (S2010[targetword])
#print (MLE2010[targetword])

#S2010.save('S2010air.bin')
#MLE2010.save('MLE2010air.bin')

#Standard2010_air = Word2Vec.load ('S2010air.bin')
#MLE2010_air = Word2Vec.load('MLE2010air.bin')


## load keyed vectors and initiate similarities
wv1 = KeyedVectors.load(corpus1+'.vec')
wv1.init_sims()
wv2 = KeyedVectors.load(corpus2+'.vec')
wv2.init_sims()

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


wv2aligned = smart_procrustes_align_gensim(wv1, wv2)
print '%s vs %s. Target word [ %s ], cosine = %.4f' % (corpus1, corpus2, targetword, scipy.spatial.distance.cosine(wv1[targetword], wv2[targetword]))
