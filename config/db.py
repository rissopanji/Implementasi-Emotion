from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from config.settings import db_name, db_url

client = MongoClient('mongodb://localhost:27017/')
db = client['tweets']

# client = MongoClient(db_url, server_api=ServerApi('1'))
# db = client.tweets

