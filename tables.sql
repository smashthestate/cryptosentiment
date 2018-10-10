CREATE TABLE users (
    id BIGSERIAL NOT NULL PRIMARY KEY,
    twitter_user_id bigint UNIQUE,
    name text,
    screen_name text,
    statuses_count int,
    followers_count int,
    friends_count int,
    location text
);

CREATE TABLE tweets (
    id BIGSERIAL NOT NULL PRIMARY KEY,
    tweet_id bigint UNIQUE,
    tweet_text text,
    user_id bigint,
    created_at timestamp with time zone,
    in_reply_to_status_id bigint,
    in_reply_to_user_id bigint,
    source text,
    retweeted boolean,
    retweet_count int,
    favorited boolean,
    favorite_count int
);

CREATE TABLE sentiments (
    id BIGSERIAL NOT NULL PRIMARY KEY,
    tweet_id bigint UNIQUE references tweets(tweet_id),
    tweet_text text,
    sentiment_naivebayes text,
    sentiment_textblob text,
    sentiment_vader text
)