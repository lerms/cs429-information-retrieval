""" Assignment 2
"""
import abc
import numpy as np


class EvaluatorFunction:
    """
    An Abstract Base Class for evaluating search results.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def evaluate(self, hits, relevant):
        """
        Do not modify.
        Params:
          hits...A list of document ids returned by the search engine, sorted
                 in descending order of relevance.
          relevant...A list of document ids that are known to be
                     relevant. Order is insignificant.
        Returns:
          A float indicating the quality of the search results, higher is better.
        """
        return


class Precision(EvaluatorFunction):

    def evaluate(self, hits, relevant):
        """
        Compute precision.

        >>> Precision().evaluate([1, 2, 3, 4], [2, 4])
        0.5
        """
        count = 0
        for doc_id in relevant:
            if doc_id in hits:
                count += 1

        return float(count / len(hits))

    def __repr__(self):
        return 'Precision'


class Recall(EvaluatorFunction):

    def evaluate(self, hits, relevant):
        """
        Compute recall.

        >>> Recall().evaluate([1, 2, 3, 4], [2, 5])
        0.5
        """
        tp = 0
        for rel in relevant:
            if rel in hits:
                tp += 1

        return float(tp / (tp + (len(relevant) - tp)))

    def __repr__(self):
        return 'Recall'


class F1(EvaluatorFunction):
    def evaluate(self, hits, relevant):
        """
        Compute F1.

        >>> F1().evaluate([1, 2, 3, 4], [2, 5])  # doctest:+ELLIPSIS
        0.333...
        """
        p = Precision.evaluate(self, hits, relevant)
        r = Recall.evaluate(self, hits, relevant)
        f1 = (2 * p * r) / (p + r)
        return f1

    def __repr__(self):
        return 'F1'


class MAP(EvaluatorFunction):
    def evaluate(self, hits, relevant):
        """
        Compute Mean Average Precision.

        >>> MAP().evaluate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 4, 6, 11, 12, 13, 14, 15, 16, 17])
        0.2
        """
        result = []
        for i in range(1, len(hits)+1):
            result.append(Precision.evaluate(self, hits[:i], relevant))
        zipped = zip(hits, result)
        mapped = []
        for each in hits:
            if each in relevant:
                mapped.append(each)
        return sum([y for x, y in zipped if x in mapped]) / len(hits)


    def __repr__(self):
        return 'MAP'

