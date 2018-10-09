-- Copy data from other db into temp table tweets_test first

WITH new_tweets as (SELECT tweets_test.tweet_id, tweets_test.tweet_text, tweets_test.user_id, tweets_test.created_at, tweets_test.in_reply_to_status_id, tweets_test.in_reply_to_user_id, tweets_test.source, tweets_test.retweeted, tweets_test.retweet_count, tweets_test.favorited, tweets_test.favorite_count
	  from tweets RIGHT JOIN tweets_test ON tweets.tweet_id = tweets_test.tweet_id WHERE tweets.id is null)

INSERT INTO tweets (tweet_id, tweet_text, user_id, created_at, in_reply_to_status_id, in_reply_to_user_id, source, retweeted, retweet_count, favorited, favorite_count) SELECT * FROM new_tweets;
