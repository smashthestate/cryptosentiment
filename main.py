import re 
import tweepy
import os
import jsonpickle
import sys
from tweepy import AppAuthHandler 
from textblob import TextBlob 
from typing import List

# App modules
import populate_db
from json_deserializer import JsonDeserializer
from models import User
from models import Tweet
import classifier

main_app_settings = {}

class TwitterClient(object):
    def __init__(self):
        with open("app_settings.json", "r") as app_settings_file:
            self.app_settings = jsonpickle.decode(app_settings_file.read())
        self.tweets_file = "tweets_4.json"
        try:
            self.auth = AppAuthHandler(self.app_settings["consumer_key"], self.app_settings["consumer_secret"])
            self.api = tweepy.API(self.auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
        except:
            print("Error: Authentication failed")

        main_app_settings["db_name"] = self.app_settings["db_name"]
        main_app_settings["db_user"] = self.app_settings["db_user"]
        main_app_settings["db_password"] = self.app_settings["db_password"]

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

    def get_tweets(self, query, latest_tweet_id, count = 10):
        # Control params
        since_id = latest_tweet_id
        max_id = 0
        max_tweets = 1000000

        tweet_count = 0
        step_count = 0

        tweets = []
        users = []
        while tweet_count < max_tweets:
            try:
                if(since_id <= 0):
                    if(max_id <= 0):
                        tweets_batch = self.api.search(q = query, count = count)
                    else:
                        tweets_batch = self.api.search(q = query, count = count, max_id = str(max_id-1))
                else:
                    if(max_id <= 0):
                        tweets_batch = self.api.search(q = query, count = count, since_id = since_id)
                    else:
                        tweets_batch = self.api.search(q = query, count = count, since_id = since_id, max_id = str(max_id-1))

                if not tweets_batch:
                    print("No more tweets found :(")
                    break

                tweet_count += len(tweets_batch)
                max_id = tweets_batch[-1].id
                step_count += 1
                print("Downloaded {0} tweets so far. Number of step: {1}".format(tweet_count, step_count))
                for fetched_tweet in tweets_batch:
                    tweet, user = self.extract_tweet_and_user(fetched_tweet)
                    user_id_list = [user.twitter_user_id for user in users]
                    if user.twitter_user_id not in user_id_list:
                        users.append(user)
                    
                    tweets.append(tweet)

            except tweepy.TweepError as e:
                print("Error: " + str(e))

        print("Downloaded {0} tweets in {1} steps".format(tweet_count, step_count))
        
        return tweets, users

    def retrieve_retweeted(self, tweets, users):
        '''
        Method checks if any of the tweets referenced in existing tweets haven't been added
        '''
        retweeted_tweets = []

        for tweet in tweets:
            if hasattr(tweet, 'retweeted_status'):
                retweeted_tweets.append(tweet.retweeted_status)

        retweeted_tweets_to_append = [tweet for tweet in retweeted_tweets if tweet not in tweets]

        user_id_list = [user.twitter_user_id for user in users]

        for fetched_tweet in retweeted_tweets_to_append:
            tweet, user = self.extract_tweet_and_user(fetched_tweet)
            tweets.append(tweet)
            if user.twitter_user_id not in user_id_list:
                users.append(user) 

        print("\nAppended {0} retweeted tweets not originally retrieved".format(len(retweeted_tweets_to_append)))
        return tweets, users

    def extract_tweet_and_user(self, fetched_tweet):
        tweet = Tweet()
        user = None
        for attribute_string in dir(fetched_tweet):
            if not attribute_string.startswith("__"):
                fetched_tweet_field = getattr(fetched_tweet, attribute_string)
                try:
                    getattr(Tweet, attribute_string)
                    setattr(tweet, attribute_string, fetched_tweet_field)
                except AttributeError as e:
                    pass

                if isinstance(fetched_tweet_field, tweepy.User) and not user:
                    user = User()
                    tweet.user_id = fetched_tweet_field.id
                    user.twitter_user_id = fetched_tweet_field.id
                    user.name = fetched_tweet_field.name
                    user.screen_name = fetched_tweet_field.screen_name
                    user.statuses_count = fetched_tweet_field.statuses_count
                    user.followers_count = fetched_tweet_field.followers_count
                    user.friends_count = fetched_tweet_field.friends_count
                    user.location = fetched_tweet_field.location

        return tweet, user

    def write_to_json(self, tweets):
        mode = "w"

        if(os.path.isfile(self.tweets_file)):
            mode = "a"

        with open(self.tweets_file, mode) as f:
            for tweet in tweets:
                f.write(jsonpickle.encode(tweet._json, unpicklable=False)+ "\n")

def main():
    api = TwitterClient()
    db_connection = populate_db.DbConnection(main_app_settings["db_name"], main_app_settings["db_user"], main_app_settings["db_password"])
    latest_tweet_id = db_connection.retrieve_latest_tweet_id()

    tweets, users = api.get_tweets(query = 'bitcoin cash', latest_tweet_id = latest_tweet_id, count = 100)
    tweets, users = api.retrieve_retweeted(tweets, users)

    db_connection.insert_tweets_into_db(tweets)
    db_connection.insert_users_into_db(users)

    # json_deserializer = JsonDeserializer("tweets")
    # tweets = json_deserializer.deserialize_json_files()
    # db_connection.insert_tweets_into_db(tweets)

    db_connection.close_connection

if __name__ == "__main__": 
    # calling main function 
    main()
