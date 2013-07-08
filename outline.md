
Classifying Tweets as political/apolitical

Demonstration of natural language processing and machine learning

-use NLTK's built-in classifier
-need code to automatically generate featuresets from tweets
-use Flask to output results in a webapp




After this step, the tweet above should look like this:

We can see that this tweet (retweet, actually) is about the recent NSA domestic surveillance scandal involving Edward Snowden. This is definitely a political tweet. In fact, we can make the safe assumption that anything Mr. Klabnik tweets about the NSA or Edward Snowden is probably political. Therefore, some of the important keywords we can use to determine that this tweet is political are "snowden," "nsa," and "whistleblower."

Some of the other words, however, are not as informative in classifying this tweet. It probably doesn't matter that this is a retweet from `@glynmoody`, for example. Words like "to" and "in 
