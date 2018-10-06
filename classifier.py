# from textblob import TextBlob
import pandas as pd
import jsonpickle
# import textblob
from langid.langid import LanguageIdentifier, model
import re
from nltk.tokenize import RegexpTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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
    sid = SentimentIntensityAnalyzer()
    regex_pattern = r'([Bb]itcoin[\s]*[Cc]ash)|(#[Bb][Cc][Hh])'
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

    for tweet_text in tweet_texts:
        if re.search(regex_pattern, tweet_text) and identifier.classify(tweet_text)[0] == 'en':
            ss = sid.polarity_scores(tweet_text)
            tweet_sentiment.append(ss['compound'])
        elif not identifier.classify(tweet_text)[0] == 'en':
            tweet_sentiment.append(-100)
        else:
            tweet_sentiment.append(-500)

        # tokenizer = RegexpTokenizer()
        # tokenizer = TweetTokenizer(strip_handles=True)
        # tweet_text = tokenizer.tokenize(tweet_text)
        # blob = TextBlob(tweet_text,tokenizer)
    
    df['sentiment_polarity'] = pd.Series(tweet_sentiment, index = df.index)

    polarity_series = df['sentiment_polarity']
    texts_high_polarity = [tweet_text for polarity, tweet_text in zip(polarity_series, df['tweet_text']) if not polarity == -500 and not polarity == 0 and not polarity == -100]

    # for polarity, text in zip(polarity_series[:100], texts_high_polarity[:100]):
    #     print("{0} has a polarity of: {1}".format(text, polarity))

    for text in texts_high_polarity[:1000]:
        print(text+"\n\n")
    
    print(len(df['sentiment_polarity']))
    print(len(texts_high_polarity))    

if __name__ == "__main__": 
    # calling main function 
    main()