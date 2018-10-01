import jsonpickle
import jsonpickle.tags as tags
import jsonpickle.unpickler as unpickler
import jsonpickle.util as util
import simplejson as json
import psycopg2
from psycopg2.extras import execute_values
import tweepy
import sys
from glob import glob

from models import User
from models import Tweet


class JsonDeserializer(object):
    def __init__(self, file_name):
        self.json_files = glob(file_name+"*.json")
        print(self.json_files)

    def deserialize_json_files(self):
        readonly_mode = "r"
        tweets = []
        for json_file in self.json_files:
            with open(json_file, readonly_mode) as f:
                # json_full_str = f.read()
                for line in f:
                    # tweet = Tweet()
                    tweet_dict = jsonpickle.decode(line)
                    tweet = self.from_json(Tweet, tweet_dict)
                    print(tweet)
                    tweets.append(tweet)
        
        return tweets

    def from_json(self, desired_class, dct):
        '''
        Method copied from:
        https://github.com/jsonpickle/jsonpickle/issues/148#issuecomment-362508753
        '''
        # object_type = str(desired_class.__class__)
        # json_dict = json.loads(json_str)
        # json_dict.update({"py/object":object_type})
        # return jsonpickle.decode(json.dumps(json_dict))

        # import jsonpickle
        # json_dict_obj = json.load(json_str)

        dct[tags.OBJECT] = util.importable_name(desired_class)
        obj = unpickler.Unpickler().restore(dct, classes=desired_class)
        return obj

    
    