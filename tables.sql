CREATE TABLE tweets (
    id BIGSERIAL NOT NULL PRIMARY KEY,
    tweet_text test,
    user_id bigint REFERENCES users,
    created_at timestamp with time zone,
    in_reply_to_status_id bigint,
    in_reply_to_user_id_str bigint,
    source text,
    retweeted boolean,
    retweet_count int,
    favorited boolean,
    favorite_count int
)

CREATE TABLE users (
    id BIGSERIAL NOT NULL PRIMARY KEY,
    name text,
    screenname text,
    statuses_count int,
    followers_count int,
    friends_count int,
    location text
)