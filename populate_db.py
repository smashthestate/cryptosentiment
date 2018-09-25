import jsonpickle
import psycopg2
import tweepy

with open("app_settings.json", "r") as app_settings_file:
    app_settings = jsonpickle.decode(app_settings_file.read())


class DbConnection(object):

    def __init__(self, db_name, db_user, db_password):
        connection_string = "dbname=" + db_name + " user=" + db_user + " password=" + db_password
        self.conn = psycopg2.connect(connection_string)

    def insert_tweets_into_db(self, tweets):
        cur = self.conn.cursor()
        insert_columns_query = ("INSERT INTO users "
                "(name, screenname, statuses_count, followers_count, friends_count, location, " # users table columns
                "tweet_text, user_id, created_at, in_reply_to_status_id, in_reply_to_user_id, " # tweets table columns
                "source, retweeted, retweet_count, favorited, favorite_count) " # tweets table columns
                "VALUES ")

        insert_user_values_query = ""

        for tweet in tweets:
            # attribute_list = [a for a in dir(tweet) if not a.startswith("__")]
            # for attribute in attribute_list:
            if tweet.user.location == "":
                tweet.user.location = " "

            insert_user_values_query += "(\"{t.user.name}\", \"{t.user.screen_name}\", {t.user.statuses_count}, {t.user.followers_count}, " \
                                    "{t.user.friends_count}, \"{t.user.location}\", \"{t.text}\", {t.user.id}, \"{t.created_at}\", " \
                                    "{t.in_reply_to_status_id}, {t.in_reply_to_user_id}, \"{t.source}\", {t.retweeted}, {t.retweet_count}, " \
                                    "{t.favorited}, {t.favorite_count}), ".format(t = tweet)
        
        if(len(tweets) > 0):
            cur.execute(insert_columns_query + insert_user_values_query)
        else:
            print("Nothing to insert!")

        self.conn.close()