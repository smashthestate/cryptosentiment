import re 
import tweepy
import os
import jsonpickle
import sys

from tweepy import AppAuthHandler 
from textblob import TextBlob 

class TwitterClient(object):

    def __init__(self):
        with open("app_settings.json", "r") as app_settings_file:
            app_settings = jsonpickle.decode(app_settings_file.read(), keys=True)
        self.file_name = "tweets_3.json"
        try:
            self.auth = AppAuthHandler(app_settings["consumer_key"], app_settings["consumer_secret"])
            self.api = tweepy.API(self.auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
        except:
            print("Error: Authentication failed")

    def clean_tweet(self, tweet): 
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split()) 

    def get_tweet_sentiment(self, tweet):
        analysis = TextBlob(self.clean_tweet(tweet))
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'

    def get_and_write_tweets(self, query, count = 10):
        tweets = []

        max_id = 0
        max_tweets = 1000000
        tweet_count = 0
        step_count = 0
        
        mode = "w"

        if(os.path.isfile(self.file_name)):
            mode = "a"

        with open(self.file_name, mode) as f:
            while tweet_count < max_tweets:
                try:
                    if(max_id <= 0):
                        fetched_tweets = self.api.search(q = query, count = count)
                    else:
                        fetched_tweets = self.api.search(q = query, count = count, max_id = str(max_id-1))
                
                    if not fetched_tweets:
                        print("No more tweets found :(")
                        break

                    tweet_count += len(fetched_tweets)
                    max_id = fetched_tweets[-1].id
                    step_count += 1
                    print("Downloaded {0} tweets so far. Number of step: {1}".format(tweet_count, step_count))
                    for tweet in fetched_tweets:
                        f.write(jsonpickle.encode(tweet._json, unpicklable=False)+ "\n")
                        parsed_tweet = {}
                        if tweet.retweet_count > 0:
                            if parsed_tweet not in tweets:
                                tweets.append(parsed_tweet)
                        else:
                            tweets.append(parsed_tweet)

                except tweepy.TweepError as e:
                    print("Error: " + str(e))

        print("Downloaded {0} tweets in {1} steps".format(tweet_count, step_count))

        return tweets

    def retrieve_write_retweeted(self, tweets):
        '''
        Function to check if any of the tweets referenced in existing tweets haven't been added
        '''
        retweeted_tweets = []

        for tweet in tweets:
            if hasattr(tweet, 'retweeted_status'):
                retweeted_tweets.append(tweet.retweeted_status)

        retweeted_tweets_to_append = [tweet for tweet in retweeted_tweets if tweet not in tweets]
        
        with open(self.file_name, "a") as f:
            for tweet in retweeted_tweets_to_append:
                tweets.append(tweet)
                f.write(jsonpickle.encode(tweet._json, unpicklable=False)+ "\n")
                print("\nTweet Id: {0}".format(tweet.id))

        print("\nAppended {0} retweeted tweets not originally retrieved".format(len(retweeted_tweets_to_append)))

def main():
    api = TwitterClient()

    tweets = api.get_and_write_tweets(query = 'bitcoin cash', count = 100)

    api.retrieve_write_retweeted(tweets)

    # ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
    # print("Positive tweets percentage: {} %".format(100*len(ptweets)/len(tweets)))

    # ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
    # print("Negative tweets percentage: {} %".format(100*len(ntweets)/len(tweets)))

    # print("Neutral tweets percentage: {} %".format(100*(len(tweets) - len(ptweets) - len(ntweets))/len(tweets)))

    # print("\n\nPositive tweets:")
    # for tweet in ptweets:
    #     tweet_text = tweet['text'].encode('utf-8')
    #     print(tweet_text)

    # print("\n\nNegative tweets:")
    # for tweet in ntweets:
    #     tweet_text = tweet['text'].encode('utf-8')
    #     print(tweet_text)

    # print("\nNumber of positive tweets:")
    # print(len(ptweets))

    # print("\nNumber of negative tweets:")
    # print(len(ntweets))

    # print("\nNumber of neutral tweets:")
    # print(len(tweets) - len(ptweets) - len(ntweets))

if __name__ == "__main__": 
    # calling main function 
    main()
