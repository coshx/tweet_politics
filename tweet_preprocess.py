# tweet_preprocess.py
# preprocess tweets before using them
# to build a featureset for a classifier

import re
import string


#utility functions for cleaning up 
#i.e., preprocessing tweet text before tokenization

def normalize_whitespace(func):
    """
    strip whitespace from the start and end of a tweet
    also, convert multiple sequential whitespace chars into one whitespace char
    """
    return lambda text: re.sub(r"[\s]{2,}", " ", func(text).strip())


def convert_to_lowercase(func):
    """
    convert tweet to all all_lowercase
    """
    return lambda text: func(text).lower()


def remove_retweets(func):
    """
    remove retweet tag ("RT") from tweet
    """
    return lambda text: re.sub(r"([\s]+|^)RT([\s]+|$)", "", func(text))


def remove_punctuation(func):
    """
    remove punctuation from text
    """
    return lambda text: func(text).translate(None, string.punctuation)


def remove_usernames(func):
    """
    remove usernames from tweet
    """
    return lambda text: re.sub(r"([\s]+|^)@[^\s]+", "", func(text))


def remove_email(func):
    """
    remove emails from tweet
    """
    return lambda text: re.sub(r"[\s]*[^@\s]+@[^@\s]+\.[^@\s]", "", func(text))


def remove_links(func):
    """
    remove hyperlinks from tweet
    """
    return lambda text: re.sub(r"[\s]*(https|http|ftp)[^\s]+", "", func(text))


def convert_to_ascii(func):
    """
    convert unicode to ascii by removing all non-ascii characters
    """
    return lambda text: func(text).encode("ascii", "ignore")


@normalize_whitespace
@convert_to_lowercase
@remove_retweets
@remove_punctuation
@remove_usernames
@remove_email
@remove_links
@convert_to_ascii
def cleanup(text):
    return text


