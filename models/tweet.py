import datetime as dt

class Tweet(object):
    class_attributes_list = ['text', 'user_id', 'created_at', 'in_reply_to_status_id', 'in_reply_to_user_id', 'source', 'retweeted', 'retweet_count', 'favorited', 'favorite_count']
    text: str = None
    user_id: int = None
    created_at: dt.datetime = None
    in_reply_to_status_id: int = None
    in_reply_to_user_id: int = None
    source: str = None
    retweeted: bool = None
    retweet_count: int = None
    favorited: bool = None
    favorite_count: int = None

    def __setattr__(self, key, value):
        if key in self.class_attributes_list:
            object.__setattr__(self, key, value)
        else:
            pass



