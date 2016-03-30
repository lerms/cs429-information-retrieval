"""
Assignment 3. Implement a Multinomial Naive Bayes classifier for spam filtering.

You'll only have to implement 3 methods below:

train: compute the word probabilities and class priors given a list of documents labeled as spam or ham.
classify: compute the predicted class label for a list of documents
evaluate: compute the accuracy of the predicted class labels.

"""

from collections import defaultdict
import glob
import math
import os


class Document(object):
    """ A Document. Do not modify.
    The instance variables are:

    filename....The path of the file for this document.
    label.......The true class label ('spam' or 'ham'), determined by whether the filename contains the string 'spmsg'
    tokens......A list of token strings.
    """

    def __init__(self, filename=None, label=None, tokens=None):
        """ Initialize a document either from a file, in which case the label
        comes from the file name, or from specified label and tokens, but not
        both.
        """
        if label:  # specify from label/tokens, for testing.
            self.label = label
            self.tokens = tokens
        else:  # specify from file.
            self.filename = filename
            self.label = 'spam' if 'spmsg' in filename else 'ham'
            self.tokenize()

    def tokenize(self):
        self.tokens = ' '.join(open(self.filename).readlines()).split()


class NaiveBayes(object):
    def __init__(self):
        self.vocab = set()                   # unique vocab set
        self.n_docs = 0                      # num of docs
        self.label_info = defaultdict(list)  # {label: [class_doc_count, [concat_text], {term: count}]}
        self.priors = defaultdict(float)     # {label: class_doc_count / total_docs}
        self.cond_prob = defaultdict(dict)   # {term: {label: probability}}

    def get_word_probability(self, label, term):
        """
        Return Pr(term|label). This is only valid after .train has been called.

        Params:
          label: class label.
          term: the term
        Returns:
          A float representing the probability of this term for the specified class.

        >>> docs = [Document(label='spam', tokens=['a', 'b']), Document(label='spam', tokens=['b', 'c']), Document(label='ham', tokens=['c', 'd'])]
        >>> nb = NaiveBayes()
        >>> nb.train(docs)
        >>> nb.get_word_probability('spam', 'a')
        0.25
        >>> nb.get_word_probability('spam', 'b')
        0.375
        """
        return self.cond_prob[term][label]

    def get_top_words(self, label, n):
        """ Return the top n words for the specified class, using the odds ratio.
        The score for term t in class c is: p(t|c) / p(t|c'), where c'!=c.

        Params:
          labels...Class label.
          n........Number of values to return.
        Returns:
          A list of (float, string) tuples, where each float is the odds ratio
          defined above, and the string is the corresponding term.  This list
          should be sorted in descending order of odds ratio.

        >>> docs = [Document(label='spam', tokens=['a', 'b']), Document(label='spam', tokens=['b', 'c']), Document(label='ham', tokens=['c', 'd'])]
        >>> nb = NaiveBayes()
        >>> nb.train(docs)
        >>> nb.get_top_words('spam', 2)
        [(2.25, 'b'), (1.5, 'a')]
        """

        zipper = defaultdict(float)
        for term in self.cond_prob:
            summed = 0
            for l in self.cond_prob[term].keys():
                if l != label:
                    summed += self.cond_prob[term][l]

            if self.cond_prob[term].get(label, False):
                zipper[term] = (self.cond_prob[term][label]) / summed
            else:
                zipper[term] = 0
        return sorted([(v, k) for (k, v) in zipper.items()], reverse=True)[:n]

    def train(self, documents):
        """
        Given a list of labeled Document objects, compute the class priors and
        word conditional probabilities, following Figure 13.2 of your
        book. Store these as instance variables, to be used by the classify
        method subsequently.
        Params:
          documents...A list of training Documents.
        Returns:
          Nothing.
        """

        self.training_doc_info(documents)  # extract all necessary info from docs
        for label in self.label_info.keys():  # ['spam', 'ham']
            self.priors[label] = float(self.label_info[label][0] / self.n_docs)
            for term in self.vocab:
                self.cond_prob[term][label] = (self.label_info[label][2][term] + 1) / \
                                              (len(self.label_info[label][1]) + len(self.vocab))
        return

    def classify(self, documents):
        """ Return a list of strings, either 'spam' or 'ham', for each document.
        Params:
          documents....A list of Document objects to be classified.
        Returns:
          A list of label strings corresponding to the predictions for each document.
        """
        score = defaultdict(float)
        labels = []
        for doc in documents:
            for label in self.label_info.keys():
                score[label] = math.log10(self.priors[label])
                for term in doc.tokens:
                    if self.cond_prob[term].get(label, False):
                        score[label] += math.log10(self.cond_prob[term][label])
            maxed = max(score, key=score.get)
            labels.append(maxed)
        return labels

    # only want to do one iteration of training docs
    def training_doc_info(self, documents):
        self.n_docs = len(documents)  # note: O(1) does not iterate to find len
        for doc in documents:
            if self.label_info.get(doc.label, False):
                self.label_info[doc.label][0] += 1
            else:
                self.label_info[doc.label] = [1, [], defaultdict(int)]
            for token in doc.tokens:
                self.vocab.add(token)
                self.label_info[doc.label][1].append(token)
                self.label_info[doc.label][2][token] += 1
        return


def evaluate(predictions, documents):
    """ Evaluate the accuracy of a set of predictions.
    Return a tuple of three values (X, Y, Z) where
    X = percent of documents classified correctly
    Y = number of ham documents incorrectly classified as spam
    X = number of spam documents incorrectly classified as ham

    Params:
      predictions....list of document labels predicted by a classifier.
      documents......list of Document objects, with known labels.
    Returns:
      Tuple of three floats, defined above.
    """

    outcomes = defaultdict(list)  # {label : [total, correct]}
    for i, doc in enumerate(documents):
        if outcomes.get(documents[i].label, False):
            outcomes[documents[i].label][0] += 1
        else:
            outcomes[documents[i].label] = [1, 0]

        if predictions[i] == documents[i].label:
            outcomes[documents[i].label][1] += 1

    return (float((outcomes['ham'][1] + outcomes['spam'][1]) / (outcomes['ham'][0] + outcomes['spam'][0]))), \
            float(outcomes['ham'][0] - outcomes['ham'][1]), \
            float(outcomes['spam'][0] - outcomes['spam'][1])


def main():
    """ Do not modify. """
    if not os.path.exists('train'):  # download data
        from urllib.request import urlretrieve
        import tarfile
        urlretrieve('http://cs.iit.edu/~culotta/cs429/lingspam.tgz', 'lingspam.tgz')
        tar = tarfile.open('lingspam.tgz')
        tar.extractall()
        tar.close()
    train_docs = [Document(filename=f) for f in glob.glob("train/*.txt")]
    print('read', len(train_docs), 'training documents.')
    nb = NaiveBayes()
    nb.train(train_docs)
    test_docs = [Document(filename=f) for f in glob.glob("test/*.txt")]
    print('read', len(test_docs), 'testing documents.')
    predictions = nb.classify(test_docs)
    results = evaluate(predictions, test_docs)
    print('accuracy=%.3f, %d false spam, %d missed spam' % (results[0], results[1], results[2]))
    print('top ham terms: %s' % ' '.join('%.2f/%s' % (v, t) for v, t in nb.get_top_words('ham', 10)))
    print('top spam terms: %s' % ' '.join('%.2f/%s' % (v, t) for v, t in nb.get_top_words('spam', 10)))


if __name__ == '__main__':
    main()
