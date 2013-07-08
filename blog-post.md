A First Dive into Natural Language Processing
=============================================


My last project as a summer intern at Coshx Labs Charlottesville involved a very exciting and very new topic for me: natural language processing. I've always been a reader, so I was happy to marry in a project my interest in computer science with my interest with language. I've spent the last couple weeks reading textbooks, watching Khan Academy and OpenCourseWare videos, and in general feeling inadequate about my math skills as I learned needed background material in probability, information theory, and linguistics. I love learning new things though, so I really can't complain.

To test what I've been learning, my boss, Ben, gave me a fun little side project: build a small web app that automatically classifies tweets as political or apolitical. While at first glance this was a fairly simple project, there's actually a lot going on. The app had to:

* **Fetch tweets using Twitter's API.** This was probably the easiest step, as there are [many][tweepy] [Python][python-twitter] [libraries][TweetPony] that make interfacing with Twitter relatively painless. I did have to learn how to use [OAuth][], though.
* **Extract information about tweets.** This was the hardest step and the bulk of the programming work, as unfortunately classifiers can't just read tweets, run them through some formulas, and output a classification; the tweets must be heavily preprocessed beforehand. In other words, classifiers don't take raw textual data as input, but rather *metadata* that somehow describes that data. (In machine learning parlance, *features* is the preferred term over *metadata*).
* **Classify the tweets using the features extracted.** As an avid Python programmer, I had heard of the [Natural Language Toolkit][NLTK] library, and I was excited that I would finally have the chance to use it. Using NLTK meant that I didn't actually have to write a classifier myself; I just had to feed a training corpus (i.e., a set of tweets that are already classified) to one of NLTK's built-in classifiers and let it do the hard math for me. The downside to this approach (usually called *supervised machine learning*) was that I had to manually classify tweets to build the corpus -- and a lot of tweets at that, if I were to build a robust corpus that captures the breadth of language used by Twitter users.
* **Display the results.** All the web app has to do was display a tweet and its classification, so the app really only had to be one page. My natural impulse was to use [Django][], but using an MVC framework seemed overkill to display a single page; instead, I opted to use the [Flask][] microframework.

To simplify the project, instead of aiming to classify *any* tweet as political or apolitical, Ben suggested that I focus on a specific Twitter user. He recommended to use Steve Klabnik's [account][steveklabnik], as Mr. Klabnik is a prolific tweeter, and most of his tweets fall into the two broad categories of programming and politics. (Ben, being a Rails developer, is also a big fan).

Since Python is my programming language of choice, the code in this article is written in Python, and I extensively use Python libraries in my examples. With that said, while you're probably not going to get as much out of this article if don't know Python, you'll still learn something from the read.


## Fetching tweets

First, we have to fetch some tweets to build a classified training corpus. Before reading this section, however, make sure you have the proper authentication keys to use Twitter's API. The [Twitter Developers][] webpage should have everything you need to know.

As I've said above, there are many Python libraries for interfacing with Twitter, so take your pick. I used [python-twitter][] for this project, but even if you use another library, the interface should be similar enough that you can follow along easily. 

Let's get to it. Fire up the Python interpreter in the command line:


```python
>> import twitter
>> api = twitter.Api(
    consumer_key="YOUR KEY",
    consumer_secret="YOUR SECRET",
    access_token_key="YOUR TOKEN",
    access_token_secret="YOUR TOKEN SECRET")
```


The `api` object, as its name implies, is the main interface to the Twitter API. Its constructor takes OAuth credentials as its arguments. Of course, replace the fake credentials with your own.

Now that you are authenticated, let's fetch some tweets.


```python
>> tweets = api.GetUserTimeline(screen_name="steveklabnik", count=200)
```


Note that 200 is the maximum number of tweets you can fetch in one method call, as set by Twitter's API. The training corpus should have more than 200 tweets, so keep calling `GetUserTimeline` and appending the `tweets` list:


```python
>> tweets += api.GetUserTimeline(screen_name="steveklabnik", count=200,
   max_id=tweets[-1].id)
```


The `max_id` argument tells `GetUserTimeline` to only fetch tweets that were posted earlier than a specific tweet. Since `tweets[-1]` gives the earliest tweet that is currently in the `tweets` list, the code snippet above appends the list with an earlier batch of tweets. Keep running this snippet until you think you have enough tweets to build a training corpus. For reference, my corpus had about ~3,000 tweets -- around the maximum number that the Twitter API allows you to fetch.

If you peek into a tweet object, you should get something like this:


```python
>> print tweets[0].text
>> u'Bug is found in the Ecuador Embassy in London http://t.co/HdhqZcTkCZ'
>> print tweets[0].id
>> 352353936361000960
>> print tweets[0].created_at
>> u'Wed Jul 03 09:11:39 +0000 2013'
```


Now we want to save this data into a serialized format so we can work on it in the future without having to download it from Twitter over and over again. You can use Python's `pickle` module or an XML library to do this; I used JSON myself.

First, let's convert the tweet objects into small dictionaries, since tweet objects have a lot of extra data that we don't really need:


```python
>> import json
>> json_tweets = [{"id": tweet.id, "text": tweet.text,
   "political": False} for tweet in tweets]
```


Now let's save the JSON object to a file:

    
```python
>> with open("steveklabnik_tweets.txt", "w") as f:
>>     json.dump(json_tweets, f)
```


Note that I added a `political` key in the tweet dictionaries, which is the political/apolitical classification for that tweet. Since you have to manually build the corpus, you have to change this for every tweet. Yes it sounds painful to classify ~3,000 tweets by hand, I know. To make the work easier on me, I wrote a script that loads the JSON file of serialized tweets, displays the tweets one by one and prompts me to classify the tweet as political/apolitical, and then serializes the updated tweet dictionaries back to the JSON file. I've posted it in a [gist][classify-manual] for your perusal.

Now that we have a corpus on disk, we can set that aside and work on the next part of the project: feature extraction.


## Extracting tweet features

From what I've experienced in this project, feature extraction is an idiosyncratic process -- that is, the kind of features that you need largely depend on the project domain. In this case, I hope that people working with tweets -- in general, not just classifying whether tweets are political/apolitical -- would find the following section informative. Note, however, what I did here came from hands-on experiments; there's not much math involved here. Some might be annoyed at the lack of rigor, but note that this was a small project with a short timeframe for completion.

For this project, I focused on extracting keywords from tweets to build a feature set. These keywords could then be used by the classifier to determine whether a tweet is political or apolitical. For example, a tweet that contains the word "government" or "senate" is probably political. The basic strategy is this: find keywords, rank the importance of individual keywords, and then create a correlation between a keyword and a classification. Finding and ranking keywords is the main goal of feature extraction; creating a correlation between keywords and classifications is the main goal of classifying the tweets. We will focus on finding and ranking keywords for this section.

For this keyword-centric approach to be effective, however, the tweets must be preprocessed heavily. Broadly speaking, we need to do the following:

* Clean up the raw text of a tweet
* Tokenize the cleaned text
* Rank importance of a token/keyword in a tweet, for all the tweets in the corpus


### Cleaning up raw text

Tweets are notoriously unstructured and that can pose a challenge for building informative feature sets. A tweet with the word "goverment" might not be classified as political, but we can readily tell that if it was spelled correctly -- "government" -- there would be a higher probability that it would be classified as political. Tweets can also have a lot of extraneous text that probably would not be useful for determining whether they are political. Take this one, for example:


> RT @bob: Hope you had a blast! Here is a link to Sheryl's pics: http://t.co/skj2o Don't forget my email bob@gmail.com #TeamCarribean ☺


It probably doesn't matter that this is a retweet from `@bob`. Words like "is" and "to" contain little semantic content and serve nothing more as syntactic scaffolding. Given this, the two main goals of cleaning up raw text are to 1) normalize the text and 2) remove extraneous text.

Again I note that feature extraction is an idiosyncratic process, so see the following as more of a guideline than a precise algorithm.

Here is how I cleaned up the text of a tweet. These steps are sequential; that is, the input of a step is the output of the previous step. If you want to see the code I wrote, see [this gist][preprocess].

1. **Convert Unicode to ASCII.** This is probably a bad idea for less trivial projects -- don't do this if you're working in a language with a non-latin script of course, like Chinese -- but just for this project we'll just convert tweets into ASCII strings and ignore non-ASCII characters to make things simple. For example, after this step, the tweet above would be would be:


    > RT @bob: Hope you had a blast! Here is a link to Sheryl's pics: http://t.co/skj2o Don't forget my email bob@gmail.com #TeamCarribean


2. **Remove links.** The text of a link is pretty useless, so we need to remove it. Now the tweet looks like this:


    > RT @bob: Hope you had a blast! Here is a link to Sheryl's pics: Don't forget my email bob@gmail.com #TeamCarribean


3. **Remove email.** This step is similar to removing links. Now the tweet looks like this:


    > RT @bob: Hope you had a blast! Here is a link to Sheryl's pics: Don't forget my email #TeamCarribean


4. **Remove usernames.** Usernames are not semantically important, so we'll remove them as well:


    > RT Hope you had a blast! Here is a link to Sheryl's pics: Don't forget my email #TeamCarribean


5. **Remove retweets.** Again, it is not imporant if a tweet is a retweet or not, so we'll remove the "RT" in front of the tweet:


    > Hope you had a blast! Here is a link to Sheryl's pics: Don't forget my email #TeamCarribean


6. **Convert hashtags.** Hashtags usually contain good semantic content, so it's important that we normalize them to be usable as keywords. A common pattern of hashtags is where a phrase is written in camel case; that is, the separate words of the hashtag are capitalized, but there are no spaces in between. It would be useful then to separate the conjoined words, so that each can be used as a possible keyword. There is also the common pattern where the words aren't capitalized (e.g., `#winninglife` versus `#WinningLife` or `#winningLife`), but for now we'll ignore that since it's much harder to separate words for that type of hashtag. Now the tweet looks like:


    > Hope you had a blast! Here is a link to Sheryl's pics: Don't forget my email Team Carribean


7. **Convert to lowercase.** Converting all words to lowercase is an easy way to normalize the text. The words "Government" and "government," for example, should be treated the same. Now the tweet looks like:


    > hope you had a blast! here is a link to sheryl's pics: don't forget my email team carribean


8. **Remove possessives.** Again, this is another normalizing step. There's not much difference from "Snowden" to "Snowden's," for example. Now the tweet looks like:


    > hope you had a blast! here is a link to sheryl pics: don't forget my email team carribean


9. **Remove punctuation.** Punctuation is for human readers and is not really informative to a classifier, so let's remove it. Now the tweet looks like:


    > hope you had a blast here is a link to sheryl pics dont forget my email team carribean


10. **Normalize whitespace.** This step removes all whitespace trailing and preceding the text, and replaces all sequence of spaces (i.e., two or more) with one space. This step simplifies tokenization later. The tweet will look the same after this step since its whitespace is normalized anyway.

Notice that there's still a lot of words -- "to," "in," "is" -- that are semantically inconsequential. Those words will be removed in the next step: tokenization.


### Tokenizing cleaned text

Tokenization basically means to split the text into a list of tokens -- which, in this case, are analogous to words. Python's NLTK library makes this process trivial, as it already has several built-in tokenizers that we can use. For this project, I used the whitespace tokenizer, which, as its name implies, splits the text according to whitespace. The other tokenizers mostly differ in how they tokenize punctuation, but since we've already removed all the punctuation from the text, we can just split the text by spaces. Tokenization code looks something like this:


```python
>> from nltk.tokenize import WhitespaceTokenizer
>> tokenizer = WhitespaceTokenizer()
>> tokenizer.tokenize(cleaned_text)
>> print cleaned_text
>> ["hope", "you", "had", "a", "blast", "here", "is", "a", "link", "to",
    "sheryl", "pics", "dont", "forget", "my", "email", "is",
    "team", "carribean"]
```


We can further clean up this list of tokens. As with cleaning up the raw text, this process is more of an art than a science, so just use the following as a guideline, not as a precise algorithm.

1. **Expand contractions.** This is another normalization step, and a fairly self-explanatory one at that: we take a contraction token and split it into its constituent tokens. Now the list of tokens looks like:


    ```python
    ["hope", "you", "had", "a", "blast", "here", "is", "a", "link", "to",
     "sheryl", "pics", "do", "not", "forget", "my", "email", "is", "team",
     "carribean"]
    ```


2. **Remove stopwords.** Remember those words like "to" and "in" that don't have much semantic content? They are actually called "stopwords," and, as you've probably guessed, we're going to remove them ([here][stopwords] is a pretty exhaustive, but by no means canonical, list of common English stopwords). Now the tweet looks like this:


    ```python
    ["hope", "blast", "link", "sheryl", "pics", "forget", "email",
    "team", "carribean"]
    ```


    This is a good, informative list of keywords. We could stop here, but I added one more step.

3. **Remove words with irrelevant parts-of-speech.** We can use one of NLTK's built-in parts of speech taggers to tag each token and then remove those tokens with irrelevant parts of speech. For example, words that are articles, like "the" or "a," unequivocally have little semantic content, so we can remove all articles from the token list. In this case, I opted to remove any token that is not a noun, verb, or adjective. Again, you can change what parts of speech you deem irrelevant. The tweet above would look the same after this step, since the removing the stopwords removed all of the tokens that had irrelevant parts of speech (at least, what I deemed irrelevant).

This concludes the tokenization step, and preprocessing in general. Finally, we can now rank keywords.


### Ranking keywords

Undoubtedly there's a lot of formulas for calculating the importance of keywords; for this project, I used a popular formula called `tf-idf` (short for *term frequency - inverse document frequency*). *Document* is the formal term, but basically I mean *tweet*. A word's tf-idf score increases as it occurs more frequently in a document, but at the same time its score decreases as it occurs more frequently in the corpus in general. This makes common words that occur in many documents have a low score for each document in which they occur, and allows rare words to have high scores for the documents that contain such words.

The mathematical details behind tf-idf are pretty understandable, but I'm not going to talk about them here. [Wikipedia][tf-idf], as always, is an invaluable resource to learn about the subject. It's a common formula and there's a myriad of implementations for it, so you can [probably][tfidflib1] [find][tfidflib2] [something][tfidflib3] on Google quite easily. I wrote a tf-idf library for my own learning; you can find it in a gist [here][preprocess]. 

I must add an important detail about my implementation of tf-idf for this particular project. Since tweets are very short (they're limited to 140 characters, as you probably know), the likelihood of a keyword appearing more than once in a tweet is very unlikely. Therefore, instead of checking how many times a tweet occurs in a document, I just checked whether the word occurs in the tweet at all. (That is, instead of using the raw frequency of a word to calculate its "tf" component score, I used the "Boolean" frequency).

What does the calculation of tf-idf scores actually look like? Let's see an example, using the list of tokens from the previous example:


```python
{
    "hope": 5.4513960434139594,
    "blast": 6.83769040453385,
    "link": 5.045930935305795,
    "sheryl": 7.243155512642015,
    "pics": 7.936302693201959,
    "forget": 5.990392544146647,
    "email": 4.758248862854014,
    "team": 5.856861151522124,
    "carribean": 7.936302693201959
}
```


The dictionary above is the feature set of the tweet we saw earlier. It's basically a list of keywords and a score that ranks how important the keyword is to the tweet. How do we calculate this? Recall that tf-idf ranks the importance of a word by how common it is relative to the tweet *and* how common it is relative to the corpus. This means that before we can calculate tf-idf scores for a single tweet, we need to gather a lot of tweets beforehand and calculate the "idf" component score for words (much as the "tf" component score is a measure of the commonality of a word relative to a single tweet, the "idf" component score is a measure of the commonality of a word relative to a corpus). This is where the corpus we saved to disk earlier comes into play. Load it from the JSON file:


```python
>> import json
>> with open("steveklabnik_tweets.txt", "r") as f:
>>     tweets = json.load(f)
```


By now it is obvious that feature extraction is a very involved process, so it would be prudent to create a simple interface that ties together all the steps we have discussed, from preprocessing to calculating tf-idf scores. To this end, I wrote a class that does just this; you can take a peek at it in [this gist][preprocess]. Using this feature set builder class looks something like this:


```python
>> from tweet_featureset import TweetFeatureset
>> tf = TweetFeatureset(tweets)
>> features = tf.build_featureset(new_tweets)
```


As you can see, the constructor takes as its argument the corpus needed to calculate the idf scores. After that, you can then build a feature set for a new tweet. Note that `build_featureset` takes as its argument a *list* of tweets, and returns a *list* of feature sets.

Now we have the ability to extract features from tweets. With this, we can finally start to classify tweets.


## Classifying tweets

As I have previously mentioned, we don't need to actually build a classifier; instead, we'll use one of NLTK's built-in classifiers and feed it a training corpus. Using this training corpus, the classifier automatically correlates certain keywords with a certain classification. NLTK has three types of built-in classifiers; for this project, I used the [naive Bayes classifier][naive-bayes].

First off, let's create a training corpus. This corpus consists of *tagged feature sets*, or feature sets coupled with a classification. This is where the hard work of classifying all those tweets come in. To build this corpus, we can use the feature set builder we used previously:


```python
>> tagged_features = tf.build_tagged_featureset(tweets)
```


Note that `build_tagged_featureset` looks for a "political" key for each tweet dictionary in the `tweets` list; that is, you must already have previously classified the tweet as political/apolitical.

If you peek inside `tagged_features`, you can see that it's nothing more than a tuple, with the first element being a feature set and the second element being a classification:


```python
>> print tagged_features[0]
>> ({'days': 4.717426868333759,
    'docs': 5.633717600207914,
    'house': 6.32686478076786,
    'laptop': 6.32686478076786,
    'nsa': 3.8932514253674095,
    'rio': 6.5500083320820695,
    'stolen': 7.243155512642015},
    True)
```


With a training corpus at hand, let's create a classifier:


```python
>> from nltk import NaiveBayesClassifier
>> classifier = NaiveBayesClassifier.train(tagged_features)
```


Voila! It's that easy using NLTK, though granted building a feature set took a while. You can now use the classifier to automatically classify tweets. Let's see it in action:


```python
>> tweet = { "text": u"RT @bob: Hope you had a blast! Here is a " +
        "link to Sheryl's pics: http://t.co/skj2o Don't forget my email " +
        "bob@gmail.com #TeamCarribean ☺"}
>> features = tf.build_featureset([tweet])[0]
>> print classifier.classify(features)
>> False
```


Note that it's common practice to split a manually tagged corpus into two: a set of tweets to use as a training corpus, and a set of tweets to use as a test corpus. The test corpus assesses the accuracy of a classifier. Thus instead of feeding *all* of the tweets in a manually classified corpus, it's more common to do the following:


```python
>> train_set, test_set = tagged_features[:2500], tagged_featureset[2500:]
>> classifier = NaiveBayesClassifier.train(train_set)
>> print nltk.classify.accuracy(test_set)
>> 0.806539509537
```


As you can see, the test corpus is *not* fed into the classifier, only the training corpus. We then test the accuracy of the classifier by having it classify tweets it hasn't seen before and comparing its classification with the classification we manually tagged. 80% accuracy is not bad -- that's a classification mismatch once every five tweets.

If you're curious about what keywords are a good signifier for political tweets, here's a list generated by a handy classifier method:


```python
>> print classifier.show_most_informative_features()
>> Most Informative Features
    nsa = 3.8932514253674095 True : False = 198.4 : 1.0
    prism = 4.80080847727281 True : False = 44.0 : 1.0
    police = 4.891780255478537 True : False = 39.7 : 1.0
    capitalism = 5.633717600207914 True : False = 30.4 : 1.0
    americans = 5.73907811586574 True : False = 26.8 : 1.0
    spying = 5.73907811586574 True : False = 26.8 : 1.0
    libertarians = 5.73907811586574 True : False  = 26.8 : 1.0
    court = 5.73907811586574 True : False = 26.8 : 1.0
    riot = 5.856861151522124 True : False = 23.2 : 1.0
    secret = 5.990392544146647 True : False = 19.7 : 1.0
```


Given that, at the time of this writing, the NSA PRISM domestic surveillance scandal is still broiling, it is no surprise that "nsa" and "prism" are the top keywords to identify a political tweet. Of course, the problem is that once this scandal wanes from the public conversation -- and therefore from Mr. Klabnik's Twitter feed -- the classifier would start to get less and less accurate. One possible way to alleviate this is to expand the corpus to include older tweets and so that the classifier can pick up important keywords that are agnostic to current events and yet still good signifiers for a political tweet.

So that's pretty much it. The only topic I didn't discuss is writing a simple Flask web app to display the results of the classifier, but that's a bit off topic -- and I'm sure you, dear reader, already have your preferred Node.js / Rails/Haskell/Assembly/etc. web framework to use and would rather not be bored with a primer to web development.

Of course, you can use the approach I've discussed here to classify distinctions other than political/apolitical. In fact, all of the code here (except for when methods look for a "political" key in a dictionary) are agnostic to what kind of classifications you are making; the classifications only show up in the training corpus you feed the classifier. Change the classifications of the tweets in the training corpus, and the classifier will happily follow.

If you are curious about what the webapp I wrote looks like, you can find it here: <http://steveklabnik-tweets-politics.herokuapp.com>


## Further reading

If you're interested in learning more about NLP, here's some resources that I found useful:

* The [NLTK Book][nltk-book]. This is not only a great tutorial for learning NLTK, it is also a great primer for natural language processing. The book focuses more on practice than theory, so it is great for those who want to build NLP programs quickly. Note that if you have you are already a proficient Python programmer, some of the chapters on Python essentials might not be useful, as the authors do not assume you have prior programming experience.
* Michael Collins's [Coursera class on NLP][coursera-nlp]. Granted, I only took a week's worth of material from this class, so I cannot say much about it, but from the material it covered, I can tell you that this class is very math intensive. It's best that you learn a bit of probability and some information theory first before taking this class; this is a math-intensive class.
* [Foundations of Statistical Natural Language Processing][foundations-snlp] by Christopher D. Manning. I've only read the introductory chapters that cover probability, information theory, and linguistics; I suggest having a strong background in probability and statistics before reading this, because Manning breezes through a lot of mathematical concepts. I spent a lot of time puzzling over the formula-intensive chapters, but that's probably because I don't have the needed background in math.
* [Khan Academy videos][khan-academy] on Information Theory. A basic introduction to information theory. It's pretty easy to follow, though it does make some diversions into history, so you might want to skip some parts.


[NLTK]: http://nltk.org "NLTK"
[OAuth]: http://oauth.net/ "OAuth"
[tweepy]: https://github.com/tweepy/tweepy "tweepy"
[python-twitter]: https://github.com/bear/python-twitter "python-twitter"
[TweetPony]: https://github.com/Mezgrman/TweetPony "TweetPony"
[Django]: https://www.djangoproject.com "Django"
[Flask]: http://flask.pocoo.org/ "Flask"
[steveklabnik]: https://twitter.com/steveklabnik "Steve Klabnik"
[Twitter Developers]: https://dev.twitter.com/ "Twitter Developers"
[python-twitter]: https://github.com/bear/python-twitter "python-twitter"
[classify-manual]: https://gist.github.com/rolph-recto/5922373 "classify-manual"
[tf-idf]: http://en.wikipedia.org/wiki/Tf-idf "tf-idf"
[preprocess]: https://gist.github.com/rolph-recto/5921554 "preprocess"
[stopwords]: http://www.ranks.nl/resources/stopwords.html "stopwords"
[tfidflib1]: http://code.google.com/p/tfidf/
[tfidflib2]: https://github.com/hrs/python-tf-idf
[tfidflib3]: https://github.com/timtrueman/tf-idf
[naive-bayes]: http://en.wikipedia.org/wiki/Naive_Bayes_classifier
[nltk-book]: http://nltk.org/book/ "NLTK Book"
[coursera-nlp]: https://class.coursera.org/nlangp-001/class/index "Coursera NLP"
[khan-academy]: https://www.khanacademy.org/math/applied-math/informationtheory/info-theory/v/intro-information-theory
[foundations-snlp]: http://www.amazon.com/Foundations-Statistical-Natural-Language-Processing/dp/0262133601/ref=sr_1_1?ie=UTF8&qid=1373033122&sr=8-1&keywords=foundations+of+natural+language+processing