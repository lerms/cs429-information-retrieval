"""
Assignment 5: K-Means. See the instructions to complete the methods below.
"""

from collections import Counter, defaultdict
import gzip
import math
import numpy as np


class Cluster(object):

    def __init__(self, num):
        self.num = num
        self.docs = list()
        self.count = 0
        self.er = 0

    def add_doc(self, doc):
        self.docs.append(doc)
        self.er += doc[1]
        self.count += 1

    def __eq__(self, other):
        return self.num == other.num

    def __str__(self):
        return 'CLUSTER ' + str(self.num)


class KMeans(object):

    def __init__(self, k=2):
        """ Initialize a k-means clusterer. Should not have to change this."""
        self.k = k
        self.er = 0
        self.clusters = []
        self.mean_vectors = []

    def cluster(self, documents, iters=10):
        """
        Cluster a list of unlabeled documents, using iters iterations of k-means.
        Initialize the k mean vectors to be the first k documents provided.
        After each iteration, print:
        - the number of documents in each cluster
        - the error rate (the total Euclidean distance between each document and its assigned mean vector), rounded to 2 decimal places.
        See Log.txt for expected output.
        The order of operations is:
        1) initialize means
        2) Loop
          2a) compute_clusters
          2b) compute_means
          2c) print sizes and error
        """

        for i in range(0, self.k):
            self.mean_vectors.append((documents[i], self.sqnorm(documents[i])))

        for i in range(0, iters):
            self.compute_clusters(documents)
            self.compute_means()
            self.error()
            print([cluster.count for cluster in self.clusters])
            print("%.2f" % self.er)

    def compute_means(self):
        """ Compute the mean vectors for each cluster (results stored in an
        instance variable of your choosing)."""

        # self.cluster_length = [len(cluster) for cluster in self.clusters]
        self.er = 0
        for cluster in self.clusters:
            mean_vector = Counter()
            for doc, distance in cluster.docs:
                mean_vector.update(doc)
            for doc in mean_vector:
                mean_vector[doc] = float(mean_vector[doc]) / float(self.clusters[cluster.num].count)
            self.mean_vectors[cluster.num] = (mean_vector, self.sqnorm(mean_vector))

    def compute_clusters(self, documents):
        """ Assign each document to a cluster. (Results stored in an instance
        variable of your choosing). """

        self.clusters = []
        for i in range(0, self.k):
            self.clusters.append(Cluster(i))

        for doc in documents:
            distances = []
            for i in range(0, self.k):
                mean = self.mean_vectors[i][0]
                mean_norm = self.mean_vectors[i][1]
                distances.append((i, self.distance(doc, mean, mean_norm)))
            distances = sorted(distances, key=lambda x: x[1])
            cluster_num = distances[0][0]  # min cluster
            distance = 1. * distances[0][1]  # min distance
            self.clusters[cluster_num].add_doc((doc, distance))

    def sqnorm(self, doc):
        """ Return the vector length of a dictionary d, defined as the sum of
        the squared values in this dict. """
        norm = 0.0
        for term in doc:
            norm += float(doc[term]) ** 2
        return norm

    def distance(self, doc, mean, mean_norm):
        """ Return the Euclidean distance between a document and a mean vector.
        See here for a more efficient way to compute:
        http://en.wikipedia.org/wiki/Cosine_similarity#Properties"""
        sq_norm = self.sqnorm(doc)
        to_sqr = (mean_norm + sq_norm) - 2.0 * dot_product(doc, mean)
        return math.sqrt(to_sqr)

    def error(self):
        """ Return the error of the current clustering, defined as the total
        Euclidean distance between each document and its assigned mean vector."""
        self.er = 0
        for cluster in self.clusters:
            mean_norm = self.mean_vectors[cluster.num][1]
            for doc, distance in cluster.docs:
                self.er += self.distance(doc, self.mean_vectors[cluster.num][0], mean_norm)

    def print_top_docs(self, n=10):
        """ Print the top n documents from each cluster. These are the
        documents that are the closest to the mean vector of each cluster.
        Since we store each document as a Counter object, just print the keys
        for each Counter (sorted alphabetically).
        Note: To make the output more interesting, only print documents with more than 3 distinct terms.
        See Log.txt for an example."""

        for cluster in self.clusters:
            print(cluster)
            to_print = [(doc, distance) for doc, distance in cluster.docs if len(doc) > 3]
            to_print = sorted(to_print, key=lambda x: x[1])
            for j in range(0, n):
                print(' '.join(sorted(to_print[j][0].keys())))


def dot_product(doc, mean):
    dot_prod = 0.0
    for term in doc:
        dot_prod += float(doc[term]) * float(mean[term])
    return dot_prod


def prune_terms(docs, min_df=3):
    """ Remove terms that don't occur in at least min_df different
    documents. Return a list of Counters. Omit documents that are empty after
    pruning words.
    >>> prune_terms([{'a': 1, 'b': 10}, {'a': 1}, {'c': 1}], min_df=2)
    [Counter({'a': 1}), Counter({'a': 1})]
    """

    doc_freq = defaultdict(int)
    for doc in docs:
        seen = set()
        for term in doc:
            if term not in seen:
                doc_freq[term] += 1
                seen.add(term)

    result_counters = []
    for doc in docs:
        valid = defaultdict(int)
        for term in doc:
            if doc_freq[term] >= min_df:
                valid[term] = doc[term]
        if len(valid) > 0:
            result_counters.append(Counter(valid))

    return result_counters


def read_profiles(filename):
    """ Read profiles into a list of Counter objects.
    DO NOT MODIFY"""
    profiles = []
    with gzip.open(filename, mode='rt', encoding='utf8') as infile:
        for line in infile:
            profiles.append(Counter(line.split()))
    return profiles


def main():
    profiles = read_profiles('profiles.txt.gz')
    print('read', len(profiles), 'profiles.')
    profiles = prune_terms(profiles, min_df=2)
    km = KMeans(k=10)
    km.cluster(profiles, iters=20)
    km.print_top_docs()


if __name__ == '__main__':
    main()