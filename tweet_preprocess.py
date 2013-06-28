# tweet_preprocess.py
# preprocess tweets before using them
# to build a featureset for a classifier

import re
import string

import nltk
from nltk.tag import pos_tag

from stopwords import stopwords
from contractions import contractions

#utility functions for cleaning up 
#i.e., preprocessing tweet text before tokenization

def normalize_whitespace(func):
    """
    strip whitespace from the start and end of a tweet
    also, convert multiple sequential whitespace chars into one whitespace char
    """
    return lambda text: re.sub(r"[\s]{2,}", " ", func(text).strip())


def remove_punctuation(func):
    """
    remove punctuation from text
    """
    #TO DO
    #preserve apostrophes between letters for contractions
    #strip possessives
    return lambda text: re.sub(r"[`~!@#$%^&*()-=_+,./<>?;':\"\[\]{}\|]",
        "", func(text))


def remove_possessives(func):
    """
    remove possessives from a noun
    """
    return lambda text: re.sub(r"[^\s]'s([\s]+|$)",
            lambda match: (match.group(0).strip()[:-2] + " "), func(text))


def convert_to_lowercase(func):
    """
    convert tweet to all all_lowercase
    """
    return lambda text: func(text).lower()


def convert_hashtags(func):
    """
    convert hashtags into individual words
    ex. #TeamJacob would be converted to team jacob
    """
    return lambda text: \
        re.sub(r"#([^\s#]+)",
        lambda match: \
            " "
            + re.sub(r"([a-z])([A-Z])", "\g<1> \g<2>", match.group(1))
            + " ",
        func(text))


def remove_retweets(func):
    """
    remove retweet tag ("RT") from tweet
    """
    return lambda text: re.sub(r"([\s]+|^)RT([\s]+|$)", " ", func(text))


def remove_usernames(func):
    """
    remove usernames from tweet
    """
    return lambda text: re.sub(r"([\s]+|^)@[^\s]+", " ", func(text))


def remove_email(func):
    """
    remove emails from tweet
    """
    return lambda text: re.sub(r"[\s]*[^@\s]+@[^@\s]+\.[^@\s]", " ", func(text))


def remove_links(func):
    """
    remove hyperlinks from tweet
    """
    return lambda text: re.sub(r"[\s]*(https|http|ftp)[^\s]+", " ", func(text))


def convert_to_ascii(func):
    """
    convert unicode to ascii by removing all non-ascii characters
    """
    return lambda text: func(text).encode("ascii", "ignore")


@normalize_whitespace
@remove_punctuation
@remove_possessives
@convert_to_lowercase
@convert_hashtags
@remove_retweets
@remove_usernames
@remove_email
@remove_links
@convert_to_ascii
def cleanup_text(text):
    return text


#preprocess tokens

def remove_irrelevant_pos_tokens(tokens):
    """
    remove tokens of a certain part of speech that is probably irrelevant
    to the political content of the tweet

    assume that only nouns, verbs, and adjectives are relevant;
    remove all other tokens
    """
    pos_tokens = pos_tag(tokens)

    return [token for token, tag in pos_tokens
        if token.startswith("N")
        or token.startswith("V")
        or token.startswith("J")
    ]


def remove_short_tokens(tokens):
    """
    remove tokens that are 2 or less characters
    we can assume that these tokens aren't important
    """
    return [token for token in tokens if len(token) > 2]


def remove_stopwords(tokens):
    """
    remove stopwords (i.e., "scaffold words" in English w/o much meaning)
    """
    global stopwords
    return [token for token in tokens if not token in stopwords]


def expand_contraction_tokens(tokens):
    """
    expand a list of tokens
    """
    global contractions
    result_tokens = []
    for token in tokens:
        is_contraction = False
        for contraction, expansion in contractions:
            if token == contraction or token == contraction.replace("'", ""):
                result_tokens += expansion.split()
                is_contraction = True
                break

        if not is_contraction:
            result_tokens.append(token)

    return result_tokens


def cleanup_tokens(tokens):
    """
    preprocess tokens before using them to build a featureset
    """
    tokens = expand_contraction_tokens(tokens)
    tokens = remove_stopwords(tokens)
    tokens = remove_short_tokens(tokens)

    return tokens