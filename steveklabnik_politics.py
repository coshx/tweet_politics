# steveklabnik_politics.py
# webapp that detects if steveklabnik is tweeting about politics

import pickle
import os
import json
from urllib2 import urlopen

import twitter
from nltk import NaiveBayesClassifier
from flask import Flask, render_template
import pylibmc
from boto.s3.connection import S3Connection

from tweet_featureset import TweetFeatureset


DEBUG = True if os.getenv("STEVEKLABNIK_TWEETS_POLITICS_DEBUG") == "True" else False
AWS_BUCKET = "steveklabnik-tweets-politics"
AWS_HOST = "s3-us-west-2.amazonaws.com"
TRAINING_CORPUS_FILE = "steveklabnik_tweets.txt"
CLASSIFIER_FILE = "classifier.txt"
FEATURESET_FILE = "featureset.txt"

#OAuth credentials
TWITTER_CONSUMER_KEY = os.getenv("TWITTER_CONSUMER_KEY")
TWITTER_CONSUMER_SECRET = os.getenv("TWITTER_CONSUMER_SECRET")
TWITTER_OAUTH_TOKEN = os.getenv("TWITTER_OAUTH_TOKEN")
TWITTER_OAUTH_TOKEN_SECRET = os.getenv("TWITTER_OAUTH_TOKEN_SECRET")
TWITTER_USER = "steveklabnik"
TWITTER_OEMBED_API_URL = "https://api.twitter.com/1/statuses/oembed.json?" \
+ "align=center" \
+ "&maxwidth=500" \
+ "&hide_media=false" \
+ "&hide_thread=false" \
+ "&id="

#cache settings
CACHE_TIMEOUT = 60 * 5 #timeout after 5 minutes

#number of users calling a classification wrong before it is fixed
WRONG_CLASSIFICATION_THRESHOLD = 10
#switch for retraining classifier
RETRAIN = True if os.getenv("STEVEKLABNIK_TWEETS_POLITICS_RETRAIN") == "True" else False
#are the corpus/classifier/featureset files local (i.e., not in AWS)?
LOCAL_FILES = DEBUG or (not RETRAIN)

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


def get_aws_connection():
    return S3Connection(
        os.getenv("AWS_ACCESS_KEY_ID"),
        os.getenv("AWS_SECRET_ACCESS_KEY"),
        host=AWS_HOST
    )


def load_corpus(conn=None):
    """
    load training corpus from file
    """
    if LOCAL_FILES:
        with open(TRAINING_CORPUS_FILE, "r") as f:
            corpus = json.load(f)

        return corpus

    else:
        bucket = conn.get_bucket(AWS_BUCKET)
        key = bucket.get_key(TRAINING_CORPUS_FILE)
        raw = key.get_contents_as_string()
        corpus = json.loads(raw)

        return corpus


def dump_corpus(corpus, conn=None):
    """
    write training corpus to a file
    """
    if LOCAL_FILES:
        with open(TRAINING_CORPUS_FILE, "w") as f:
            json.dump(corpus, f)

    else:
        bucket = conn.get_bucket(AWS_BUCKET)
        key = bucket.get_key(TRAINING_CORPUS_FILE)
        key.set_contents_from_string(json.dumps(corpus))


def load_classifier(conn=None):
    """
    load classifier from a pickle file
    """
    if LOCAL_FILES:
        with open(CLASSIFIER_FILE, "rb") as f:
            classifier = pickle.load(f)

        return classifier

    else:
        bucket = conn.get_bucket(AWS_BUCKET)
        key = bucket.get_key(CLASSIFIER_FILE)
        raw = key.get_contents_as_string()
        classifier = pickle.loads(raw)

        return classifier


def dump_classifier(classifier, conn=None):
    """
    serialize/pickle classifier to a file
    """
    if LOCAL_FILES:
        with open(CLASSIFIER_FILE, "wb") as f:
            pickle.dump(classifier, f)

    else:
        bucket = conn.get_bucket(AWS_BUCKET)
        key = bucket.get_key(CLASSIFIER_FILE)
        key.set_contents_from_string(pickle.dumps(classifier))


def load_featureset(conn=None):
    """
    load featureset builder from pickle file
    """
    if LOCAL_FILES:
        with open(FEATURESET_FILE, "rb") as f:
            featureset = pickle.load(f)

        return featureset

    else:
        bucket = conn.get_bucket(AWS_BUCKET)
        key = bucket.get_key(FEATURESET_FILE)
        raw = key.get_contents_as_string()
        featureset = pickle.loads(raw)

        return featureset


def dump_featureset(tf, conn=None):
    """
    serialize/pickle featureset to a file
    """
    if LOCAL_FILES:
        with open(FEATURESET_FILE, "wb") as f:
            pickle.dump(tf, f)

    else:
        bucket = conn.get_bucket(AWS_BUCKET)
        key = bucket.get_key(FEATURESET_FILE)
        key.set_contents_from_string(pickle.dumps(tf))


def get_tweet_display(tweet_id):
    """
    Use Twitter's oEmbed API to retrieve a HTML snippet of the tweet
    """
    url = urlopen(TWITTER_OEMBED_API_URL + str(tweet_id))
    data = json.load(url)

    return data["html"]


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
        #fetch last tweet
        tweet = get_raw_tweet()

        if LOCAL_FILES:
            classifier = load_classifier()
            tf = load_featureset()

        else:
            try:
                conn = get_aws_connection()
                #load classifier and featureset
                classifier = load_classifier(conn)
                tf = load_featureset(conn)
            finally:
                conn.close()

        #classify tweet
        tweet_features = tf.build_featureset([tweet])[0]
        tweet["political"] = classifier.classify(tweet_features)

        #get HTML display of tweet
        tweet["html"] = get_tweet_display(tweet["id"])

        #save tweet to the cache
        cache.set("tweet", tweet, time=CACHE_TIMEOUT)

        return tweet
    #if the tweet is in the cache, serve that
    else:
        return cache["tweet"]


@app.route("/")
def index():
    #fetch classified tweet
    tweet = get_classified_tweet()

    return render_template("index.html", tweet=tweet)


def retrain_classifier(tweet):
    """
    add a tweet to the classifier's training corpus and retrain it
    """
    #add the tweet to the training corpus
    if LOCAL_FILES:
        corpus = load_corpus()

    else:
        try:
            conn = get_aws_connection()
            corpus = load_corpus(conn)
        finally:
            pass

    corpus.append({
        "id": tweet["id"],
        "text": tweet["text"],
        "political": tweet["political"]
    })

    #create a new featureset builder and overwrite the existing one
    tf = TweetFeatureset(corpus)
    #build a featureset to train the classifier
    train_set = tf.build_tagged_featureset(corpus)
    #retrain the classifier and save it
    classifier = NaiveBayesClassifier.train(train_set)

    #update files
    if LOCAL_FILES:
        dump_corpus(corpus)
        dump_featureset(tf)
        dump_classifier(classifier)

    else:
        try:
            dump_corpus(corpus, conn)
            dump_featureset(tf, conn)
            dump_classifier(classifier, conn)
        finally:
            conn.close()


@app.route("/wrong")
def wrong_classification():
    """
    load this page when a user thinks a tweet is incorrectly classified
    """
    global cache
    #assume that there is a cached tweet, or else don't do anything
    #correct the tweet ONLY ONCE

    if RETRAIN and "tweet" in cache and not "corrected" in cache["tweet"]:
        if not "wrong" in cache:
            cache.set("wrong", 0, time=0)
        
        cache.set("wrong", cache["wrong"] + 1)

        #if there are enough users that think the classification is wrong,
        #change it and insert the correct classification
        #into the training corpus of the classifier
        if cache.get("wrong") >= WRONG_CLASSIFICATION_THRESHOLD:
            cache.set("wrong", 0)
            tweet = cache["tweet"]
            tweet["corrected"] = True
            tweet["political"] = not tweet["political"]
            cache.set("tweet", tweet)

            retrain_classifier(tweet)
            changed = True

    return render_template("wrong.html",
        retrain=RETRAIN,
        corrected="corrected" in cache["tweet"]
    )


if __name__ == "__main__":
    app.run()