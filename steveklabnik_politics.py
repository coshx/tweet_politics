# steveklabnik_politics.py
# webapp that detects if steveklabnik is tweeting about politics

import pickle
import os

import twitter
from nltk import NaiveBayesClassifier
from flask import Flask, render_template
import pylibmc

from tweet_featureset import TweetFeatureset


DEBUG = True if os.getenv("STEVEKLABNIK_TWEETS_POLITICS_DEBUG") == "True" else False
CLASSIFIER_FILE = "classifier.txt"
FEATURESET_FILE = "featureset.txt"

#OAuth credentials
TWITTER_CONSUMER_KEY = os.getenv("TWITTER_CONSUMER_KEY")
TWITTER_CONSUMER_SECRET = os.getenv("TWITTER_CONSUMER_SECRET")
TWITTER_OAUTH_TOKEN = os.getenv("TWITTER_OAUTH_TOKEN")
TWITTER_OAUTH_TOKEN_SECRET = os.getenv("TWITTER_OAUTH_TOKEN_SECRET")
TWITTER_USER = "steveklabnik"

#cache settings
CACHE_TIMEOUT = 60 * 5 #timeout after 5 minutes

#initialize Flask app
app = Flask(__name__)
app.config.from_object(__name__)

#initialize memcached
if DEBUG:
    #development settings
    cache = pylibmc.Client(
        servers=["127.0.0.1"],
        binary=True
    )
else:
    #production settings
    cache = pylibmc.Client(
        servers=[os.environ.get('MEMCACHIER_SERVERS')],
        username=os.environ.get('MEMCACHIER_USERNAME'),
        password=os.environ.get('MEMCACHIER_PASSWORD'),
        binary=True
    )


def get_raw_tweet():
    """
    fetch the latest tweet of TWITTER_USER (steveklabnik)
    """
    api = twitter.Api(
        consumer_key=TWITTER_CONSUMER_KEY,
        consumer_secret=TWITTER_CONSUMER_SECRET,
        access_token_key=TWITTER_OAUTH_TOKEN,
        access_token_secret=TWITTER_OAUTH_TOKEN_SECRET
    )

    tweets = api.GetUserTimeline(
        screen_name=TWITTER_USER,
        trim_user=True,
        count=1
    )
    tweet = tweets[0]

    return {
        "id": tweet.id,
        "text": tweet.text,
        "retweet": tweet.retweeted_status != None
    }


def load_classifier():
    """
    load classifier from a pickle file
    """
    with open(CLASSIFIER_FILE, "rb") as f:
        classifier = pickle.load(f)

    return classifier


def load_featureset():
    """
    load featureset builder from pickle file
    """
    with open(FEATURESET_FILE, "rb") as f:
        featureset = pickle.load(f)

    return featureset


def get_classified_tweet():
    """
    load a tweet and classify it
    """
    global cache
    #check if the last tweet is in the cache
    #if not, fetch it
    #caching would create the possibility that the tweet displayed
    #is not the last tweet, but given small enough values CACHE_TIMEOUT,
    #this problem would be a decent sacrifice for performance
    if not "tweet" in cache:
        print "That shit is not cashed"
        #fetch last tweet
        tweet = get_raw_tweet()

        #load classifier and featureset
        classifier = load_classifier()
        tf = load_featureset()

        #classify tweet
        tweet_features = tf.build_featureset([tweet])[0]
        tweet["political"] = classifier.classify(tweet_features)

        #save tweet to the cache
        cache.set("tweet", tweet, time=CACHE_TIMEOUT)

        return tweet
    #if the tweet is in the cache, serve that
    else:
        print "That shit is cashed"
        return cache.get("tweet")


@app.route("/")
def index():
    #fetch classified tweet
    tweet = get_classified_tweet()

    return render_template("index.html", tweet=tweet)


if __name__ == "__main__":
    app.run()