# from textblob import TextBlob
import pandas as pd
import jsonpickle
# import textblob
from langid.langid import LanguageIdentifier, model
import re
import nltk.classify.util
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import TweetTokenizer
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import twitter_samples
from nltk.corpus.reader.twitter import TwitterCorpusReader
# from nltk.corpus.reader import twitter

from populate_db import DbConnection
from json_deserializer import JsonDeserializer

tokenizer = TweetTokenizer(strip_handles=True)

def word_feats(text):
    words = tokenizer.tokenize(text)
    return dict([(word, True) for word in words])

def main():
    app_settings = ''

    with open("app_settings.json", "r") as app_settings_file:
        app_settings = jsonpickle.decode(app_settings_file.read())

    db = DbConnection(app_settings["db_name"], app_settings["db_user"], app_settings["db_password"])
    connection = db.conn

    db_table = "SELECT * FROM tweets"

    tweets_df = pd.read_sql_query(db_table, connection)

    tweet_texts = tweets_df["tweet_text"]
    regex_pattern = r'([Bb]itcoin[\s]*[Cc]ash)|(#[Bb][Cc][Hh])'
    tweet_sentiment = []
    # sid = SentimentIntensityAnalyzer()
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

    nltk_positive_tweets_json = twitter_samples.docs(fileids='positive_tweets.json')
    nltk_negative_tweets_json = twitter_samples.docs(fileids='negative_tweets.json')

    posfeats = [(word_feats(positive_tweet['text']), 'pos') for positive_tweet in nltk_positive_tweets_json]
    negfeats = [(word_feats(negative_tweet['text']), 'neg') for negative_tweet in nltk_negative_tweets_json]

    pos_cutoff = (int)(len(posfeats)*3/4)
    neg_cutoff = (int)(len(negfeats)*3/4)

    nltk_train_data = negfeats[:neg_cutoff] + posfeats[:pos_cutoff]
    nltk_test_data = negfeats[neg_cutoff:] + posfeats[pos_cutoff:]

    classifier = NaiveBayesClassifier.train(nltk_train_data)
    print('Train data consists of {0} tweets, test data consists of {1} tweets'.format(len(nltk_train_data), len(nltk_test_data)))
    print('Accuracy yo: {0}'.format(nltk.classify.util.accuracy(classifier, nltk_test_data)))
    classifier.show_most_informative_features()

    for tweet_text in tweet_texts:
            label = classifier.classify(word_feats(tweet_text))
            tweet_sentiment.append(label)
        # if re.search(regex_pattern, tweet_text) and identifier.classify(tweet_text)[0] == 'en':
        #     ss = sid.polarity_scores(tweet_text)
        #     tweet_sentiment.append()
        # elif not identifier.classify(tweet_text)[0] == 'en':
        #     tweet_sentiment.append(-100)
        # else:
        #     tweet_sentiment.append(-500)

    #     # tokenizer = RegexpTokenizer()
    #     # tokenizer = TweetTokenizer(strip_handles=True)
    #     # tweet_text = tokenizer.tokenize(tweet_text)
    #     # blob = TextBlob(tweet_text,tokenizer)

    tweet_sentiment_series = pd.Series(tweet_sentiment, name='sentiment')
    tweets_df = pd.concat([tweets_df, tweet_sentiment_series], axis=1)
    tweets_df['sentiment'].fillna('none')
    # Refactor column names part!
    db.insert_df_into_table('sentiments', 'tweet_id, tweet_text, sentiment', tweets_df.loc[:,['tweet_id','tweet_text','sentiment']])
    # polarity_series = df['sentiment_polarity']
    # texts_high_polarity = [tweet_text for polarity, tweet_text in zip(polarity_series, df['tweet_text']) if not polarity == -500 and not polarity == 0 and not polarity == -100]

    # for polarity, text in zip(polarity_series[:100], texts_high_polarity[:100]):
    #     print("{0} has a polarity of: {1}".format(text, polarity))

    # for text, label in zip(df['tweet_text'],df['sentiment_polarity']):
    #     print(text + " " + label + "\n\n")
    

    # print(len(df['sentiment_polarity']))
    # print(len(texts_high_polarity))    

if __name__ == "__main__": 
    # calling main function 
    main()