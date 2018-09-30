import datetime as dt

class User(object):
    twitter_user_id: int = None
    name: str = None
    screen_name: str = None
    statuses_count: int = None
    followers_count: int = None
    friends_count: int = None
    location: str = None