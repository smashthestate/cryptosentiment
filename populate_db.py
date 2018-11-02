import jsonpickle
import psycopg2
from psycopg2.extras import execute_values
import tweepy
from typing import List
import datetime as dt
import re

from models import User
from models import Tweet

with open("app_settings.json", "r") as app_settings_file:
    app_settings = jsonpickle.decode(app_settings_file.read())

class DbConnection(object):

    def __init__(self, db_name, db_user, db_password):
        connection_string = "dbname=" + db_name + " user=" + db_user + " password=" + db_password
        self.conn = psycopg2.connect(connection_string)

    def insert_tweets_into_db(self, tweets: List[Tweet]):
        cur = self.conn.cursor()
        insert_tweets_query = ("INSERT INTO tweets_full "
            "(tweet_id, tweet_text, user_id, created_at, in_reply_to_status_id, in_reply_to_user_id, "
            "source, retweeted, retweet_count, favorited, favorite_count) "
            "VALUES %s "
            "ON CONFLICT (tweet_id) DO UPDATE "
            "SET retweeted = EXCLUDED.retweeted, "
            "SET retweet_count = EXCLUDED.retweet_count, "
            "SET favorited = EXCLUDED.favorited, "
            "SET favorite_count = EXCLUDED.favorite_count "
            "RETURNING id")
        
        insert_tweets_values = []

        for tweet in tweets:
            insert_tweets_values.append((tweet.id, tweet.full_text, tweet.user_id, tweet.created_at,
                                    tweet.in_reply_to_status_id, tweet.in_reply_to_user_id, tweet.source,
                                    tweet.retweeted, tweet.retweet_count,
                                    tweet.favorited, tweet.favorite_count))

        if(len(tweets) > 0):
            execute_values(cur, insert_tweets_query, insert_tweets_values)
        else:
            print("Nothing to insert!")

        self.conn.commit()

    def insert_users_into_db(self, users: List[User]):
        cur = self.conn.cursor()
        insert_users_query = ("INSERT INTO users_full "
                "(twitter_user_id, name, screen_name, statuses_count, followers_count, friends_count, location) "
                "VALUES %s "
                "ON CONFLICT (twitter_user_id) DO UPDATE "
                "SET name = EXCLUDED.name, "
                "SET screen_name = EXCLUDED.screen_name, "
                "SET statuses_count = EXCLUDED.statuses_count, "
                "SET followers_count = EXCLUDED.followers_count, "
                "SET friends_count = EXCLUDED.friends_count, "
                "SET location = EXCLUDED.location"
                "RETURNING id")

        insert_users_values = []

        for user in users:
            insert_users_values.append((user.twitter_user_id, user.name, user.screen_name, user.statuses_count, user.followers_count,
                            user.friends_count, user.location))

        if(len(users) > 0):
            execute_values(cur, insert_users_query, insert_users_values)
        else:
            print("Nothing to insert!")

        self.conn.commit()

    def retrieve_latest_tweet_id(self):
        cur = self.conn.cursor()
        query_select_latest = "SELECT tweet_id FROM tweets ORDER BY created_at DESC LIMIT 1"
        cur.execute(query_select_latest)
        latest_tweet_id = cur.fetchone()[0]
        return latest_tweet_id

    def retrieve_tweets(self):
        cur = self.conn.cursor()
        query_select_tweets = "SELECT tweet_id FROM tweets"
        cur.execute(query_select_tweets)
        tweets = cur.fetchall()
        return tweets

    def insert_df_into_table(self, table, columns, dataframe):
        cur = self.conn.cursor()
        # columns = [column, ','.join(column) for column in columns]
        query = "INSERT INTO {} ({}) VALUES %s".format(table, columns).replace("[","").replace("]", "")
        query = re.sub("[\[\]']", "", query) 
        query += "ON CONFLICT (tweet_id) DO NOTHING"
        # (" + ",".join(['{}']*len(columns)) + ")
        # query = query.format(table, columns)

        data_list = [row for row in dataframe.itertuples(index=False, name=None)]

        execute_values(cur, query, data_list)
        self.conn.commit()

    def close_connection(self):
        self.conn.close()

