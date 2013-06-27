# classify_politics.py
# manually classify tweets as political or not


import sys
import json


if __name__ == "__main__":
    #read tweets from JSON file
    tweets = []
    with open(sys.argv[1], "r") as f:
        tweets = json.load(f)

    #see if user specified a range of tweets to classify
    begin = 0
    end = len(tweets)
    if len(sys.argv) >= 4:
        begin = int(sys.argv[2])
        end = int(sys.argv[3])

    #iterate through each tweet and classify it as political/apolitical
    for tweet in tweets[begin:end]:
        print "ID: {0} \n {1}".format(tweet["id"], tweet["text"].encode("ascii", "replace"))

        #ask the user to classify the tweet
        while True:
            input = raw_input("Is this a political tweet (y/n)?")
            if input == "y":
                tweet["political"] = True
                break
            elif input == "n":
                tweet["political"] = False
                break

    #write back the tweets into the JSON file
    with open(sys.argv[1], "w") as f:
        tweets = json.dump(tweets, f)
