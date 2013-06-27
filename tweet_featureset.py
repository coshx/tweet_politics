# tweet_featureset.py
# build a featureset from a tweet corpus

# a tweet corpus is assumed to be a list of tweets,
# where a tweet is a dictionary with at least the following keys:
# -id
# -text
# -political (Boolean; answers whether a tweet is political or not)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from tweet_preprocess import cleanup
from tf_idf import tf_idf_corpus

stopwords_english = stopwords.words("english")

def tokenize_corpus(corpus):
    """
    return a tweet corpus in tokenized form
    """
    global stopwords_english

    for tweet in corpus:
        #cleanup tweets
        tweet["text"] = cleanup(tweet["text"])
        #tokenize tweets using NLTK word tokenizer
        tweet["tokens"] = word_tokenize(tweet["text"])
        #remove stopwords
        tweet["tokens"] = [
            token
            for token in tweet["tokens"]
            if not token in stopwords_english
        ]

    #remove empty tweets from corpus
    return [tweet for tweet in corpus if len(tweet["tokens"]) > 0]


def tweet_featureset(corpus):
    """
    build a featureset for a classifier using a tweet corpus
    """

    #tokenize corpus
    corpus = tokenize_corpus(corpus)

    #extract features from tweet corpus
    token_corpus = [tweet["tokens"] for tweet in corpus]
    feature_corpus = tf_idf_corpus(token_corpus)

    #pair features with respective tags
    tagged_features = [
        (features, corpus[i]["political"])
        for i, features in enumerate(feature_corpus)
    ]

    return tagged_features