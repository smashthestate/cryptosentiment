import jsonpickle
import psycopg2
from psycopg2.extras import execute_values
import tweepy
from typing import List
import datetime as dt

from models import User
from models import Tweet

with open("app_settings.json", "r") as app_settings_file:
    app_settings = jsonpickle.decode(app_settings_file.read())

class DbConnection(object):

    def __init__(self, db_name, db_user, db_password):
        connection_string = "dbname=" + db_name + " user=" + db_user + " password=" + db_password
        self.conn = psycopg2.connect(connection_string)

    def insert_tweets_into_db(self, tweets: List[Tweet], users: List[User]):
        cur = self.conn.cursor()
        insert_users_query = ("INSERT INTO users "
                "(twitter_user_id, name, screen_name, statuses_count, followers_count, friends_count, location) "
                "VALUES %s "
                "ON CONFLICT (twitter_user_id) DO NOTHING")

        insert_tweets_query = ("INSERT INTO tweets "
            "(tweet_id, tweet_text, user_id, created_at, in_reply_to_status_id, in_reply_to_user_id, "
            "source, retweeted, retweet_count, favorited, favorite_count) "
            "VALUES %s "
            "ON CONFLICT (tweet_id) DO NOTHING")
        
        insert_tweets_values = []
        insert_users_values = []

        for tweet in tweets:
            insert_tweets_values.append((tweet.id, tweet.text, tweet.user_id, tweet.created_at,
                                    tweet.in_reply_to_status_id, tweet.in_reply_to_user_id, tweet.source,
                                    tweet.retweeted, tweet.retweet_count,
                                    tweet.favorited, tweet.favorite_count))

        for user in users:
            insert_users_values.append((user.twitter_user_id, user.name, user.screen_name, user.statuses_count, user.followers_count,
                            user.friends_count, user.location))

        if(len(tweets) > 0):
            execute_values(cur, insert_users_query, insert_users_values)
            execute_values(cur, insert_tweets_query, insert_tweets_values)
        else:
            print("Nothing to insert!")

        self.conn.commit()

    def retrieve_latest_tweet_id(self):
        self.cur = self.conn.cursor()

        query_select_latest = "SELECT tweet_id FROM tweets ORDER BY created_at DESC LIMIT 1"
        self.cur.execute(query_select_latest)
        latest_tweet_id = self.cur.fetchone()[0]
        
        # tweet_id is 2nd column
        # latest_tweet_id = tweet[1]

        return latest_tweet_id

    def close_connection(self):
        self.conn.close()

