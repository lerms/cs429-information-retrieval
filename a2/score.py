""" Assignment 2
"""
import abc
from collections import defaultdict
import math
import index


def idf(term, index):
    """ Compute the inverse document frequency of a term according to the
    index. IDF(T) = log10(N / df_t), where N is the total number of documents
    in the index and df_t is the total number of documents that contain term
    t.

    Params:
      terms....A string representing a term.
      index....A Index object.
    Returns:
      The idf value.

    >>> idx = index.Index(['a b c a', 'c d e', 'c e f'])
    >>> idf('a', idx) # doctest:+ELLIPSIS
    0.477...
    >>> idf('d', idx) # doctest:+ELLIPSIS
    0.477...
    >>> idf('e', idx) # doctest:+ELLIPSIS
    0.176...
    """

    return float(math.log10(len(index.documents) / index.doc_freqs[term]))


class ScoringFunction:
    """ An Abstract Base Class for ranking documents by relevance to a
    query. """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def score(self, query_vector, index):
        """
        Do not modify.

        Params:
          query_vector...dict mapping query term to weight.
          index..........Index object.
        """
        return


class RSV(ScoringFunction):
    """
    See lecture notes for definition of RSV.

    idf(a) = log10(3/1)
    idf(d) = log10(3/1)
    idf(e) = log10(3/2)
    >>> idx = index.Index(['a b c', 'c d e', 'c e f'])
    >>> rsv = RSV()
    >>> rsv.score({'a': 1., 'c': 3}, idx)[1]  # doctest:+ELLIPSIS
    0.4771...
    """

    def score(self, query_vector, index):

        rsvs = defaultdict(float)
        for query_term in query_vector:
            if index.index.get(query_term, False):
                for posting in index.index[query_term]:
                        rsvs[posting[0]] = idf(query_term, index)
        return rsvs

    def __repr__(self):
        return 'RSV'


class BM25(ScoringFunction):
    """
    See lecture notes for definition of BM25.

    log10(3) * (2*2) / (1(.5 + .5(4/3.333)) + 2) = log10(3) * 4 / 3.1 = .6156...
    >>> idx = index.Index(['a a b c', 'c d e', 'c e f'])
    >>> bm = BM25(k=1, b=.5)
    >>> bm.score({'a': 1.}, idx)[1]  # doctest:+ELLIPSIS
    0.61564032...
    """
    def __init__(self, k=1, b=.5):
        self.k = k
        self.b = b

    def score(self, query_vector, index):

        bm25 = defaultdict(float)
        for query_term in query_vector:
            for posting in index.index[query_term]:
                score = idf(query_term, index) * (
                    ((self.k + 1) * posting[1]) / (self.k * ((1 - self.b) +
                    (self.b * (index.doc_lengths[posting[0]]) / index.mean_doc_length) +
                    posting[1])))
                bm25[posting[0]] = score
        return bm25

    def __repr__(self):
        return 'BM25 k=%d b=%.2f' % (self.k, self.b)


class Cosine(ScoringFunction):
    """
    See lecture notes for definition of Cosine similarity.  Be sure to use the
    precomputed document norms (in index), rather than recomputing them for
    each query.

    >>> idx = index.Index(['a a b c', 'c d e', 'c e f'])
    >>> cos = Cosine()
    >>> cos.score({'a': 1.}, idx)[1]  # doctest:+ELLIPSIS
    0.792857...
    """
    def score(self, query_vector, index):

        cos_vals = defaultdict(float)
        for query_term in query_vector:
            if index.index.get(query_term, False):
                for posting in index.index[query_term]:
                    tf_idf = (1 + math.log10(posting[1])) * idf(query_term, index)
                    cos_vals[posting[0]] += query_vector[query_term] * tf_idf

        for doc_id in cos_vals:
            cos_vals[doc_id] /= index.doc_norms[doc_id]

        return cos_vals


    def __repr__(self):
        return 'Cosine'
