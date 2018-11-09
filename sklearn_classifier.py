# from textblob import TextBlob
import pandas as pd
import numpy as np
import re
import jsonpickle
import collections
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

from nltk.corpus import stopwords

from populate_db import DbConnection
from json_deserializer import JsonDeserializer

def clean_tweet(tweet):
    tweet = ' '.join(re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
    tweet = re.sub("&amp;", " ", tweet)
    tweet = re.sub("&quote;", " ", tweet)
    tweet_word_list = tweet.split(" ")
    # tweet = " ".join([word for word in tweet_word_list if word not in stopwords.words("english")])
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
    tweets_df_stanford_train = shuffle(pd.read_csv("stanford_training.csv", header=None, names=colnames, encoding="ISO-8859-1"))
    tweets_df_stanford_test = pd.read_csv("stanford_test.csv", header=None, names=colnames, encoding="ISO-8859-1")

    text_clf = Pipeline([('vect', HashingVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', SGDClassifier(penalty='l2',
                                            alpha=1e-3, random_state=42,
                                            max_iter=5, tol=None))
                        ])

    text_nb_clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer(smooth_idf=False)),
                    ('clf', MultinomialNB())
                    ])

    negative_tweets = tweets_df_stanford_train[tweets_df_stanford_train['polarity']==0]
    positive_tweets = tweets_df_stanford_train[tweets_df_stanford_train['polarity']==4]
    negative_tweets_fraction = negative_tweets[:int(0.4*len(negative_tweets))]
    
    train_data = shuffle(pd.concat([positive_tweets, negative_tweets_fraction]))

    train_data['text'] = train_data['text'].apply(clean_tweet)   
    text_nb_clf = text_nb_clf.fit(train_data['text'], train_data['polarity'])

    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                  #'tfidf__use_idf': (True,  False),
                  'clf__alpha': (1e-2, 1e-3),
    }

    non_neutral_examples_binary = tweets_df_stanford_test['polarity'] != 2 
    predicted = text_nb_clf.predict(tweets_df_stanford_test[non_neutral_examples_binary]['text'])
    print(metrics.accuracy_score(tweets_df_stanford_test[non_neutral_examples_binary]['polarity'], predicted))
    print(metrics.classification_report(tweets_df_stanford_test[non_neutral_examples_binary]['polarity'], predicted))    

    show_most_informative_features(text_nb_clf.steps[0][1], text_nb_clf.steps[2][1])
    # for param_name in sorted(parameters.keys()):
    #     print('Best {0} value: {1}'.format(param_name, gs_clf.best_params_[param_name]))


if __name__ == "__main__": 
    # calling main function 
    main()