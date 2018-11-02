# from textblob import TextBlob
import pandas as pd
import jsonpickle
import collections
# import textblob
from sklearn.naive_bayes import GaussianNB

from populate_db import DbConnection
from json_deserializer import JsonDeserializer
from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer(strip_handles=True)

def word_feats(text):
    words = tokenizer.tokenize(text)
    return dict([(word, True) for word in words])

def best_word_feats(text, bestwords):
    words = tokenizer.tokenize(text)
    return dict([(word, True) for word in words if word in bestwords])


def extract_feats(dataframe):
    posfeats = [(word_feats(positive_tweet['text']), 'pos') for index, positive_tweet in dataframe.iterrows() if positive_tweet['polarity'] == 4]
    neutfeats = [(word_feats(neutral_tweet['text']), 'neutral') for index, neutral_tweet in dataframe.iterrows() if neutral_tweet['polarity'] == 2]
    negfeats = [(word_feats(negative_tweet['text']), 'neg') for index, negative_tweet in dataframe.iterrows() if negative_tweet['polarity'] == 0]

    all_feats = posfeats + neutfeats + negfeats
    
    return all_feats

def extract_best_feats(dataframe, bestwords):
    # modify tuple for sklearn classifier
    posfeats = [(best_word_feats(positive_tweet['text'], bestwords), 'pos') for index, positive_tweet in dataframe.iterrows() if positive_tweet['polarity'] == 4]
    neutfeats = [(best_word_feats(neutral_tweet['text'], bestwords), 'neutral') for index, neutral_tweet in dataframe.iterrows() if neutral_tweet['polarity'] == 2]
    negfeats = [(best_word_feats(negative_tweet['text'], bestwords), 'neg') for index, negative_tweet in dataframe.iterrows() if negative_tweet['polarity'] == 0]

    all_feats = posfeats + neutfeats + negfeats
    
    return all_feats

def retrieve_label(polarity_score):
    if polarity_score > 0:
        label = 'pos'
    elif polarity_score < 0:
        label = 'neg'
    else:
        label = 'neutral'
    return label

def main():
    # app_settings = ''

    # with open("app_settings.json", "r") as app_settings_file:
    #     app_settings = jsonpickle.decode(app_settings_file.read())

    # db = DbConnection(app_settings["db_name"], app_settings["db_user"], app_settings["db_password"])
    # connection = db.conn

    # db_table = "SELECT * FROM tweets"

    colnames = ['polarity', 'tweet_id', 'date', 'query', 'user', 'text']
    tweets_df_stanford_train = pd.read_csv("stanford_training.csv", header=None, names=colnames, encoding="ISO-8859-1")
    tweets_df_stanford_test = pd.read_csv("stanford_test.csv", header=None, names=colnames, encoding="ISO-8859-1")

    train_feats_array = []
    train_labels_array = []

    for (feats, label) in extract_feats(tweets_df_stanford_train[1:100]):
        train_feats_array.append(feats)
        train_labels_array.append(label)

    # test_feats_array = []
    # test_labels_array = []
    
    # for (feats, label) in extract_feats(tweets_df_stanford_test):
    #     test_feats_array.append(feats)
    #     test_labels_array.append(label)

    gnb = GaussianNB()
    model = gnb.fit(train_feats_array, train_labels_array)

if __name__ == "__main__": 
    # calling main function 
    main()