# steveklabnik_politics.py
# webapp that detects if steveklabnik is tweeting about politics

import pickle
import os

import twitter
from nltk import NaiveBayesClassifier
from flask import Flask, render_template

from tweet_featureset import TweetFeatureset


DEBUG = True
CLASSIFIER_FILE = "classifier.txt"
FEATURESET_FILE = "featureset.txt"

#OAuth credentials
TWITTER_CONSUMER_KEY = os.getenv("TWITTER_CONSUMER_KEY")
TWITTER_CONSUMER_SECRET = os.getenv("TWITTER_CONSUMER_SECRET")
TWITTER_OAUTH_TOKEN = os.getenv("TWITTER_OAUTH_TOKEN")
TWITTER_OAUTH_TOKEN_SECRET = os.getenv("TWITTER_OAUTH_TOKEN_SECRET")
TWITTER_USER = "steveklabnik"

app = Flask(__name__)
app.config.from_object(__name__)


def get_tweet():
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
        exclude_replies=True,
        include_rts=False,
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


@app.route("/")
def index():
    #fetch last tweet
    tweet = get_tweet()

    #load classifier and featureset
    classifier = load_classifier()
    tf = load_featureset()

    #build context
    feature = tf.build_featureset([tweet])[0]
    is_political = classifier.classify(feature)

    return render_template("index.html",
        political=is_political,
        tweet=tweet
    )


if __name__ == "__main__":
    app.run()