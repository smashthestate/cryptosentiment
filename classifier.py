from tweepy import AppAuthHandler 
from textblob import TextBlob
import pandas as pd
import jsonpickle
import textblob

from populate_db import DbConnection

def main():
    app_settings = ''

    with open("app_settings.json", "r") as app_settings_file:
        app_settings = jsonpickle.decode(app_settings_file.read())

    db = DbConnection(app_settings["db_name"], app_settings["db_user"], app_settings["db_password"])
    connection = db.conn

    db_table = "SELECT * FROM tweets"

    df = pd.read_sql_query(db_table, connection)

    tweet_texts = df["tweet_text"]
    tweet_sentiment = []
    
    for tweet_text in tweet_texts:
        blob = TextBlob(tweet_text)
        tweet_sentiment.append(blob.sentiment.polarity)
    
    df['sentiment_polarity'] = pd.Series(tweet_sentiment, index = df.index)

    polarity_series = df['sentiment_polarity']
    texts_high_polarity = [tweet_text for polarity, tweet_text in zip(polarity_series, df['tweet_text']) if polarity > 0.5]
    print(texts_high_polarity[:50])
    

if __name__ == "__main__": 
    # calling main function 
    main()