# from textblob import TextBlob
import pandas as pd
import jsonpickle
import collections
# import textblob
from langid.langid import LanguageIdentifier, model
import re
import nltk.classify.util
from nltk.metrics import scores
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import TweetTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import twitter_samples
from nltk.corpus.reader.twitter import TwitterCorpusReader
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from textblob import TextBlob
# from nltk.corpus.reader import twitter

from populate_db import DbConnection
from json_deserializer import JsonDeserializer

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
    app_settings = ''

    with open("app_settings.json", "r") as app_settings_file:
        app_settings = jsonpickle.decode(app_settings_file.read())

    db = DbConnection(app_settings["db_name"], app_settings["db_user"], app_settings["db_password"])
    connection = db.conn

    db_table = "SELECT * FROM tweets"

    # tweets_df = pd.read_sql_query(db_table, connection)

    # Source of data: Sentiment 140
    colnames = ['polarity', 'tweet_id', 'date', 'query', 'user', 'text']
    tweets_df_stanford_train = pd.read_csv("stanford_training.csv", header=None, names=colnames, encoding="ISO-8859-1")
    tweets_df_stanford_test = pd.read_csv("stanford_test.csv", header=None, names=colnames, encoding="ISO-8859-1")

    # tweet_texts = tweets_df["tweet_text"]
    # regex_pattern = r'([Bb]itcoin[\s]*[Cc]ash)|(#[Bb][Cc][Hh])'

    tweet_sentiment_naivebayes = []
    tweet_sentiment_textblob = []
    tweet_sentiment_vader = []

    sid = SentimentIntensityAnalyzer()
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

    # nltk_positive_tweets_json = twitter_samples.docs(fileids='positive_tweets.json')
    # nltk_negative_tweets_json = twitter_samples.docs(fileids='negative_tweets.json')

    # Used for single word feats
    # nltk_train_data = extract_feats(tweets_df_stanford_train)
    # nltk_test_data = extract_feats(tweets_df_stanford_test)
    # pos_cutoff = (int)(len(posfeats)*3/4)
    # neg_cutoff = (int)(len(negfeats)*3/4)
    
    # nltk_train_data = negfeats[:neg_cutoff] + posfeats[:pos_cutoff]
    # nltk_test_data = negfeats[neg_cutoff:] + posfeats[pos_cutoff:]

    # classifier = NaiveBayesClassifier.train(nltk_train_data)

    word_fd = FreqDist()
    label_word_fd = ConditionalFreqDist()

    for index, tweet in tweets_df_stanford_train.iterrows():
        word_list = tokenizer.tokenize(tweet['text'])
        for word in word_list:
            word_fd[word.lower()] += 1
            if tweet['polarity'] == 4:
                label_word_fd['pos'][word.lower()] += 1
            elif tweet['polarity'] == 2:
                label_word_fd['neutral'][word.lower()] += 1
            else:
                label_word_fd['neg'][word.lower()] += 1

    pos_word_count = label_word_fd['pos'].N()
    neutral_word_count = label_word_fd['neutral'].N()
    neg_word_count = label_word_fd['neg'].N()

    total_word_count = pos_word_count + neutral_word_count + neg_word_count

    word_scores = {}

    for word in word_fd:
        freq = word_fd[word]
        pos_score = BigramAssocMeasures.chi_sq(label_word_fd['pos'][word],(freq, pos_word_count), total_word_count)
        neutral_score = BigramAssocMeasures.chi_sq(label_word_fd['neutral'][word],(freq, neutral_word_count+1), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(label_word_fd['neg'][word],(freq, neg_word_count), total_word_count)

        word_scores[word] = neg_score + pos_score + neutral_score 

    best = sorted(word_scores.items(), key=lambda ws: ws[1], reverse=True)[:10000]
    bestwords = set([w for w, s in best])

    nltk_train_data = extract_best_feats(tweets_df_stanford_train, bestwords)
    nltk_test_data = extract_best_feats(tweets_df_stanford_test, bestwords)

    classifier = NaiveBayesClassifier.train(nltk_train_data)

    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(nltk_test_data):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)

    print('Train data consists of {0} tweets, test data consists of {1} tweets'.format(len(nltk_train_data), len(nltk_test_data)))
    print('Accuracy yo: {0}'.format(nltk.classify.util.accuracy(classifier, nltk_test_data)))
    print('pos precision: {0}'.format(scores.precision(refsets['pos'], testsets['pos'])))
    print('pos recall: {0}'.format(scores.recall(refsets['pos'], testsets['pos'])))
    print('pos F-measure: {0}'.format(scores.f_measure(refsets['pos'], testsets['pos'])))
    print('neg precision: {0}'.format(scores.precision(refsets['neg'], testsets['neg'])))
    print('neg recall: {0}'.format(scores.recall(refsets['neg'], testsets['neg'])))
    print('neg F-measure: {0}'.format(scores.f_measure(refsets['neg'], testsets['neg'])))
    
    classifier.show_most_informative_features()

    # for tweet_text in tweet_texts:
    #     if re.search(regex_pattern, tweet_text) and identifier.classify(tweet_text)[0] == 'en':
    #         label = classifier.classify(word_feats(tweet_text))
    #         tweet_sentiment_naivebayes.append(label)

    #         tweet_textblob = TextBlob(tweet_text)
    #         label = retrieve_label(tweet_textblob.sentiment.polarity)
    #         tweet_sentiment_textblob.append(label)

    #         ss = sid.polarity_scores(tweet_text)
    #         label = retrieve_label(ss['compound'])
    #         tweet_sentiment_vader.append(label)
    #     else:
    #         label = 'none'
    #         tweet_sentiment_naivebayes.append(label)
    #         tweet_sentiment_textblob.append(label)
    #         tweet_sentiment_vader.append(label)

    #     # tokenizer = RegexpTokenizer()
    #     # tokenizer = TweetTokenizer(strip_handles=True)
    #     # tweet_text = tokenizer.tokenize(tweet_text)
    #     # blob = TextBlob(tweet_text,tokenizer)

    # tweet_sentiment_series = pd.Series(tweet_sentiment, name='sentiment')
    # tweets_df = pd.concat([tweets_df, tweet_sentiment_series], axis=1)
    # sentiments_dict = {'sentiment_naivebayes': tweet_sentiment_naivebayes, 
    #                     'sentiment_textblob': tweet_sentiment_textblob,
    #                     'sentiment_vader': tweet_sentiment_vader
    #                  }

    # tweet_sentiments_df = pd.DataFrame(sentiments_dict, tweets_df.index)
    # tweets_df = pd.concat([tweets_df, tweet_sentiments_df], axis = 1)

    # cols_to_insert = ['tweet_id', 'tweet_text', 'sentiment_naivebayes', 'sentiment_textblob', 'sentiment_vader']
    # tweets_df[cols_to_insert].to_csv("tweets.csv")

    # db.insert_df_into_table('sentiments', cols_to_insert, tweets_df.loc[:,cols_to_insert])

if __name__ == "__main__": 
    # calling main function 
    main()