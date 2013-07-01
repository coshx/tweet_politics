# tf_idf.py
# term frequency - inverse document frequency
# word ranker used to build featureset for classifiers

# for convenience, the functions below assume the following:
# a "document" is a list of tokens
# a "corpus" is a list of documents (a list of lists of tokens)

#allows for integer division that returns rational numbers, not rounded integers
from __future__ import division
import math


class IdfDict(dict):
    """
    dictionary of idf values
    """

    def __init__(self, corpus_size, *args, **kwargs):
        super(IdfDict, self).__init__(*args, **kwargs)
        self.corpus_size = corpus_size


    def __getitem__(self, key):
        #if a term is not in the dictionary, calculate an idf value anyway
        if not key in self:
            return math.log(self.corpus_size)
        else:
            return super(IdfDict, self).__getitem__(key)


def tf_raw(term, document):
    return document.count(term)


def tf_bool(term, document):
    return 1 if term in document else 0


def tf_log(term, document):
    return math.log(document.count(term))


def tf(term, document, algorithm="RAW"):
    """
    Calculates the frequency of a term within a document
    Can specific different algorithms:
    -RAW calculates the raw frequency of a term
        -i.e., number of times it occurs in the document
    -BOOL calculates the "boolean frequency" of a term
        -i.e., if the term is in the document, tf is 1; if not, tf is 0
    -LOG calculates a logarithmically scaled term frequency
        -i.e., tf = log of the raw frequency
    -AUG calculates term frequency divided by the max frequency of the word
    in the document
        -i.e., this prevents bias towards longer documents

    NOTE: only RAW, BOOL, and LOG are implemented at the moment
    """

    if algorithm == "RAW":
        return tf_raw(term, document)
    elif algorithm == "BOOL":
        return tf_bool(term, document)
    elif algorithm == "LOG":
        return tf_log(term, document)
    else:
        raise ValueError("tf cannot use algorithm %s" % algorithm)


def idf(term, corpus):
    """
    Calculates the inverse document frequency for a term
    idf is the log of the ratio between the total number of documents
    in the corpus and the number of docs in the corpus with the given term
    """
    corpus_size = len(corpus)
    docs_with_term = 0

    for document in corpus:
        if term in document:
            docs_with_term += 1

    #add 1 to docs_with_term to account for terms that don't occur in the corpus
    #so that a division by zero doesn't occur
    return math.log( corpus_size / (docs_with_term+1) )


def tf_idf(term, document, corpus, algorithm="RAW"):
    """
    return tf-idf score
    """
    #if idf score was calculated previously, don't calculate it again;
    #this occurs when the tf-idf of a term is being calculated
    #for all documents in a corpus
    if type(corpus) is float:
        return tf(term, document, algorithm) * corpus
    else:
        return tf(term, document, algorithm) * idf(term, corpus)


def idf_corpus(corpus):
    """
    calculates idf score for all terms in a corpus
    """

    #build idf score for all terms in the corpus
    #first, build a vocab of the corpus
    vocab = set()
    for document in corpus:
        vocab |= set(document)

    #then, calculate the idf for each term in the vocab
    idf_set = IdfDict(len(corpus))
    for term in vocab:
        idf_set[term] = idf(term, corpus)

    return idf_set


def tf_idf_corpus(corpus, algorithm="RAW", idf_set=None):
    """
    calculates tf-idf score for all terms in every document of a corpus
    """
    #retrieve idf scores for all words in corpus
    if idf_set == None:
        idf_set = idf_corpus(corpus)

    #calculate tf-idf score for every document
    doc_set = []
    for document in corpus:
        doc_vocab = set(document)
        tf_idf_set = {}
        #calculate tf and then tf-idf score for every term
        for term in doc_vocab:
            tf_score = tf(term, document, algorithm)
            tf_idf_set[term] = tf_idf(term, document, idf_set[term], algorithm)

        doc_set.append(tf_idf_set)

    return doc_set