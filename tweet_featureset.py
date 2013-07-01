# tweet_featureset.py
# build a featureset from a tweet corpus

# a tweet corpus is assumed to be a list of tweets,
# where a tweet is a dictionary with at least the following keys:
# -id
# -text
# -political (Boolean; answers whether a tweet is political or not)

from nltk.tokenize import WhitespaceTokenizer

from tweet_preprocess import cleanup_text, cleanup_tokens
from tf_idf import tf, idf_corpus, tf_idf_corpus


class TweetFeatureset(object):
    """
    TweetFeatureset class
    creates featuresets for tweets
    """

    tokenizer = WhitespaceTokenizer()


    def __init__(self, corpus):
        self.train(corpus)


    @classmethod
    def tokenize_tweet(cls, tweet):
        """
        tokenize a single tweet
        """
        #cleanup tweets
        tweet["text"] = cleanup_text(tweet["text"])
        #tokenize tweets using NLTK word tokenizer
        #use the whitespace tokenizer to preserve contractions,
        #which will be expanded during tokenization processing
        tweet["tokens"] = cls.tokenizer.tokenize(tweet["text"])
        #clean up tokenized tweets
        tweet["tokens"] = cleanup_tokens(tweet["tokens"])

        return tweet


    @classmethod
    def tokenize_corpus(cls, corpus):
        """
        return a tweet corpus in tokenized form
        """
        corpus = [TweetFeatureset.tokenize_tweet(tweet) for tweet in corpus]

        #remove empty tweets from corpus
        return [tweet for tweet in corpus if len(tweet["tokens"]) > 0]


    def train(self, corpus):
        """
        use corpus to calculate idf scores
        """
        corpus = TweetFeatureset.tokenize_corpus(corpus)
        token_corpus = [tweet["tokens"] for tweet in corpus]
        self.idf_set = idf_corpus(token_corpus)


    def build_featureset(self, corpus, algorithm="BOOL"):
        """
        build a featureset for a classifier using a tweet corpus
        use boolean frequency algorithm for tweets because tweets are so short
        there is no reason to count words in each tweet,
        just detect if a word is in the tweet
        """
        #tokenize corpus
        corpus = TweetFeatureset.tokenize_corpus(corpus)

        #extract features from tweet corpus
        token_corpus = [tweet["tokens"] for tweet in corpus]
        feature_corpus = tf_idf_corpus(token_corpus, algorithm, self.idf_set)

        #pair features with respective tags
        tagged_features = [
            (features, corpus[i]["political"])
            for i, features in enumerate(feature_corpus)
        ]

        return tagged_features
