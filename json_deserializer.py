import jsonpickle
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
        tweet = Tweet()
        for json_file in self.json_files:
            with open(json_file, readonly_mode) as f:
                f


    
    