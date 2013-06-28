# tweet_preprocess.py
# preprocess tweets before using them
# to build a featureset for a classifier

import re
import string
from functools import reduce

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


#list from: http://en.wikipedia.org/wiki/Wikipedia:List_of_English_contractions
contractions = [
    ("aren't", "are not"),
    ("can't", "cannot"),
    ("can't've", "cannot have"),
    ("'cause", "because"),
    ("could've", "could have"),
    ("couldn't", "could not"),
    ("couldn't've", "could not have"),
    ("didn't", "did not"),
    ("doesn't", "does not"),
    ("don't", "do not"),
    ("hadn't", "had not"),
    ("hadn't've", "had not have"),
    ("hasn't", "has not"),
    ("haven't", "have not"),
    ("he'd", "he had"),
    ("he'd've", "he would have"),
    ("he'll", "he will"),
    ("he'll've", "he will have"),
    ("he's", "he is"),
    ("how'd", "how did"),
    ("how'd'y", "how did you"),
    ("how'll", "how will"),
    ("how's", "how is"),
    ("i'd", "i had"),
    ("i'd've", "i would have"),
    ("i'll", "i will"),
    ("i'll've", "i will have"),
    ("i'm", "i am"),
    ("i've", "i have"),
    ("isn't", "is not"),
    ("it'd", "it had"),
    ("it'd've", "it would have"),
    ("it'll", "it will"),
    ("it'll've", "it will have"),
    ("it's", "it is"),
    ("let's", "let us"),
    ("ma'am", "madam"),
    ("might've", "might have"),
    ("mightn't", "might not"),
    ("mightn't've", "might not have"),
    ("must've", "must have"),
    ("mustn't", "must not"),
    ("mustn't've", "must not have"),
    ("needn't", "need not"),
    ("o'clock", "of the clock"),
    ("oughtn't", "ought not"),
    ("oughtn't've", "ought not have"),
    ("shan't", "shall not"),
    ("shan't've", "shall not have"),
    ("she'd", "she had"),
    ("she'd've", "she would have"),
    ("she'll", "she will"),
    ("she'll've", "she will have"),
    ("she's", "she is"),
    ("should've", "should have"),
    ("shouldn't", "should not"),
    ("shouldn't've", "should not have"),
    ("so's", "so is"),
    ("that's", "that is"),
    ("there'd", "there would"),
    ("there's", "there is"),
    ("they'd", "they would"),
    ("they'll", "they will"),
    ("they'll've", "they will have"),
    ("they're", "they are"),
    ("they've", "they have"),
    ("to've", "to have"),
    ("wasn't", "was not"),
    ("we'd", "we would"),
    ("we'll", "we will"),
    ("we'll've", "we will have"),
    ("we're", "we are"),
    ("we've", "we have"),
    ("weren't", "were not"),
    ("what'll", "what will"),
    ("what'll've", "what will have"),
    ("what're", "what are"),
    ("what's", "what is"),
    ("what've", "what have"),
    ("when's", "when is"),
    ("when've", "when have"),
    ("where'd", "where did"),
    ("where's", "where is"),
    ("where've", "where have"),
    ("who'll", "who will"),
    ("who'll've", "who will have"),
    ("who's", "who is"),
    ("who've", "who have"),
    ("why's", "why is"),
    ("will've", "will have"),
    ("won't", "will not"),
    ("won't've", "will not have"),
    ("would've", "would have"),
    ("wouldn't", "would not"),
    ("wouldn't've", "would not have"),
    ("y'all", "you all"),
    ("y'all'd've", "you all would have"),
    ("y'all're", "you all are"),
    ("y'all've", "you all have"),
    ("you'd", "you would"),
    ("you'd've", "you would have"),
    ("you'll", "you will"),
    ("you'll've", "you will have"),
    ("you're", "you are"),
    ("you've", "you have") 
]

def iterate_contractions(text):
    """
    iterate through contraction list
    and replace all contractions with their respective expansions
    """
    global contractions
    for contraction, expansion in contractions:
        text = text.replace(contraction, expansion)

    return text


def expand_contractions(func):
    """
    expand common contractions
    """
    return lambda text: iterate_contractions(func(text))


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
@expand_contractions
@convert_to_lowercase
@convert_hashtags
@remove_retweets
@remove_usernames
@remove_email
@remove_links
@convert_to_ascii
def cleanup(text):
    return text


