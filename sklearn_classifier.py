# from textblob import TextBlob
import pandas as pd
import numpy as np
import re
import jsonpickle
import collections
import itertools
# import textblob
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from nltk.corpus import stopwords
from nltk.collocations import *
from nltk.metrics import BigramAssocMeasures
from nltk.stem.porter import PorterStemmer

from populate_db import DbConnection
from json_deserializer import JsonDeserializer

# def bigram_word_feats(tweets, score_fn = BigramAssocMeasures.chi_sq, n=200):
#     bigram_finder = BigramCollocationFinder.from_words(tweets)
#     best_bigrams = bigram_finder.nbest(score_fn, n)
#     for tweet in tweets:
#         if best_bigrams in tweet:
#             best_bigrams

#     return best_bigrams

def clean_tweet(tweet, stopwords):
    # tweet = ' '.join(re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
    tweet = re.sub(r"(@[A-Za-z0-9]+)", " ", tweet) # Remove @mentions
    tweet = re.sub(r"(\w+://\S+)", " ", tweet) # Remove URLs
    tweet = re.sub("[0-9]+", " ", tweet) # Remove numbers
    tweet = re.sub("&amp;", "and", tweet) # Remove twitter api & representation
    tweet = re.sub("&quote;", " ", tweet) # Remove twitter api " representation
    tweet = re.sub(r"([^0-9A-Za-z \t])", " ", tweet) # Remove special characters
    # tweet = ' '.join(tweet.split())
    # tweet_word_list = tweet.split()
    # tweet = " ".join([word for word in tweet_word_list if word not in stopwords])

    # loop for removing letters repeating more than twice
    # index_counter = 0
    # for letter in tweet:
    #     if index_counter>1 and letter.lower() == tweet[index_counter-1].lower() and letter.lower() == tweet[index_counter-2].lower():
    #         tweet = tweet[:index_counter] + tweet[index_counter+1:]
    #         index_counter -= 1
    #     index_counter += 1

    return tweet

def show_most_informative_features(vectorizer, clf, n=10):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n+1):-1])
    for (coef1, fn1), (coef2, fn2) in top:
        print("Negative class feat: {0} : {1} \n".format(fn1, coef1))
        print("Positive class feat: {0} : {1} \n".format(fn2, coef2))

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

    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,2))),
                        # ('tfidf', TfidfTransformer()),
                        ('clf', SGDClassifier(penalty='l2',
                                            alpha=1e-3, random_state=42,
                                            max_iter=5, tol=None))
                        ])

    text_nb_clf = Pipeline([('vect', CountVectorizer()),
                    # ('tfidf', TfidfTransformer()),
                    ('clf', MultinomialNB())
                    ])

    text_lr_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,2))),
                    ('tfidf', TfidfTransformer()),
                    ('clf', LogisticRegression(random_state=0, solver='lbfgs'))
                    ])

    negative_tweets = tweets_df_stanford_train[tweets_df_stanford_train['polarity']==0]
    # negative_tweets = negative_tweets[-int(0.35*len(negative_tweets)):]
    positive_tweets = tweets_df_stanford_train[tweets_df_stanford_train['polarity']==4]

    # Split data by 60% : 20% : 20%
    train_neg_cutoff = int(0.6*len(negative_tweets))
    val_neg_cutoff = int(0.8*len(negative_tweets))

    negative_tweets_train = negative_tweets[:train_neg_cutoff]
    negative_tweets_val = negative_tweets[train_neg_cutoff:val_neg_cutoff]
    negative_tweets_test = negative_tweets[val_neg_cutoff:]

    train_pos_cutoff = int(0.6*len(positive_tweets))
    val_pos_cutoff = int(0.8*len(positive_tweets))

    positive_tweets_train = positive_tweets[:train_pos_cutoff]
    positive_tweets_val = positive_tweets[train_pos_cutoff:val_pos_cutoff]
    positive_tweets_test = positive_tweets[val_pos_cutoff:]

    train_data = pd.concat([positive_tweets_train, negative_tweets_train])
    val_data = pd.concat([positive_tweets_val, negative_tweets_val])
    test_data = pd.concat([positive_tweets_test, negative_tweets_test])

    stopwords_set = set()

    for word in stopwords.words("english"):
        word = re.sub("'","",word)
        stopwords_set.add(word)

    train_data['text'] = train_data['text'].apply(lambda x: clean_tweet(x, stopwords_set)) 

    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                #   'tfidf__use_idf': (True,  False),
                  'clf__alpha': (1e-2, 1e-3),
    }

    text_lr_clf = text_lr_clf.fit(train_data['text'], train_data['polarity'])

    non_neutral_examples_binary = val_data['polarity'] != 2
    predicted = text_lr_clf.predict(val_data[non_neutral_examples_binary]['text'])
    print(metrics.accuracy_score(val_data[non_neutral_examples_binary]['polarity'], predicted))
    print(metrics.classification_report(val_data[non_neutral_examples_binary]['polarity'], predicted))    
    show_most_informative_features(text_lr_clf.steps[0][1], text_lr_clf.steps[2][1])
    # for param_name in sorted(parameters.keys()):
    #     print('Best {0} value: {1}'.format(param_name, gs_text_clf.best_params_[param_name]))

if __name__ == "__main__": 
    # calling main function 
    main()