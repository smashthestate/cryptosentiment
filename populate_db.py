import jsonpickle
import psycopg2
from psycopg2.extras import execute_values
import tweepy

with open("app_settings.json", "r") as app_settings_file:
    app_settings = jsonpickle.decode(app_settings_file.read())


class DbConnection(object):

    def __init__(self, db_name, db_user, db_password):
        connection_string = "dbname=" + db_name + " user=" + db_user + " password=" + db_password
        self.conn = psycopg2.connect(connection_string)

    def insert_tweets_into_db(self, tweets):
        cur = self.conn.cursor()
        insert_users_query = ("INSERT INTO users "
                "(name, screenname, statuses_count, followers_count, friends_count, location) "
                "VALUES %s")

        insert_tweets_query = ("INSERT INTO tweets "
            "(tweet_text, user_id, created_at, in_reply_to_status_id, in_reply_to_user_id, "
            "source, retweeted, retweet_count, favorited, favorite_count) "
            "VALUES %s")
        
        insert_tweets_values = []
        insert_users_values = []

        for tweet in tweets:
            u = tweet.user
            insert_users_values.append((u.name, u.screen_name, u.statuses_count, u.followers_count,
                                    u.friends_count, u.location))
           
            insert_tweets_values.append((tweet.text, tweet.user.id, tweet.created_at,
                                    tweet.in_reply_to_status_id, tweet.in_reply_to_user_id, tweet.source,
                                    tweet.retweeted, tweet.retweet_count,
                                    tweet.favorited, tweet.favorite_count))

        if(len(tweets) > 0):
            execute_values(cur, insert_users_query, insert_users_values)
            execute_values(cur, insert_tweets_query, insert_tweets_values)
        else:
            print("Nothing to insert!")

        self.conn.commit()
        self.conn.close()