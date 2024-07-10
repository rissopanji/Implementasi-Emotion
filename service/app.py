import pickle
import os
import pandas as pd
import numpy as np
import re
import requests

from flask import Flask, request, jsonify
from config.db import db
from model.tweet import Tweet
from service.sentiment import Sentiment
from service.crawler import TweetCrawler
from dotenv import load_dotenv

load_dotenv()
auth_token = os.getenv("AUTH_TOKEN")


app = Flask(__name__)

def clean_data(data):
    for doc in data:
        for key, value in doc.items():
            if pd.isna(value):
                doc[key] = None
        if 'topic_probability':
            del doc ['topic_probability']   # atau nilai default yang sesuai
        if '__v' in doc:
            del doc['__v']
    return data

@app.route('/')
def index():
    return 'Hello World!'

@app.route('/result', methods=['GET'])
def get_result():
    try:
        keyword = request.args.get('keyword')
        jumlah_tweet = request.args.get('jumlah_tweet', default=5, type=int)
        start_date = request.args.get('start_date') # Expected format: "YYYY-MM-DD"
        end_date = request.args.get('end_date')  # Expected format: "YYYY-MM-DD"

        if not (start_date and end_date):
            return jsonify({"error": "Start date and end date must be provided"}), 400
        
        # search to database
        keyword_regex = f".*{keyword}.*"
        cursor = Tweet.getTweetsByKeyword(keyword=keyword_regex, limit=jumlah_tweet, start_date=start_date, end_date=end_date)

        # If there are no tweets in the database, start crawling
        if not cursor or len(cursor) < jumlah_tweet:
            tweet_crawler = TweetCrawler(auth_token=auth_token, 
                                         search_keyword=keyword, 
                                         limit=jumlah_tweet, 
                                         start_date=start_date, 
                                         end_date=end_date)
            
            tweet_crawler.harvest_tweets()

            path_to_file = f"./tweets-data/{keyword}.csv"
            if os.path.exists(path_to_file):
                df = pd.read_csv(path_to_file)

                # Replace all NaN values with None across the DataFrame
                df.replace(np.nan, '', inplace=True, regex=True)

                data_crawling = []
                for index, row in df.iterrows():
                    tweet_data = row.to_dict()
                    # tweet_data['_id'] = str(row['id'])  # Convert ObjectId to string
                    data_crawling.append(tweet_data)
                

                sentiment = Sentiment.classify_sentiment(data=data_crawling)
                
                # Insert the tweets into the database
                Tweet.insertTweets(data_crawling)

                return jsonify(sentiment), 200

            else:
                return jsonify({"error": "No tweets found"}), 404

        # classify sentiment
        data = []
        for tweet in cursor:
            tweet_data = tweet.copy()  # Make a copy of the dictionary to modify it
            tweet_data['_id'] = str(tweet['_id'])  # Convert ObjectId to string
            data.append(tweet_data)

        sentiment = Sentiment.classify_sentiment(data=data)

        # Update the sentiment of the tweets in the database
        Tweet.updateSentiment(sentiment)

        return jsonify(sentiment), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500     

@app.route('/sentimen-all', methods=['GET'])
def get_all_sentiment():
    try:
        cursor = Tweet.getAllTweets()

        data = []
        for tweet in cursor:
            tweet_data = tweet.copy()  # Make a copy of the dictionary to modify it
            tweet_data['_id'] = str(tweet['_id'])  # Convert ObjectId to string
            data.append(tweet_data)

        sentiment = Sentiment.classify_sentiment(data=data)

        # Update the sentiment of the tweets in the database
        Tweet.updateSentiment(sentiment)

        return jsonify(sentiment), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500    

@app.route('/crawl', methods=['GET'])
def fetch_tweets():
    auth_token = request.args.get('auth_token')
    search_keyword = request.args.get('search_keyword')
    limit = request.args.get('limit', default=10, type=int)
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    tweet_crawler = TweetCrawler(auth_token=auth_token,
                                 search_keyword=search_keyword,
                                 limit=limit,
                                 start_date=start_date,
                                 end_date=end_date)
    tweet_crawler.harvest_tweets()
    return jsonify({"success" : "crawling data success!"}),200

@app.route('/sentimen', methods=['GET'])
def classify_sentiment():
    try:
        keyword = request.args.get('keyword', 'moist cosrx')
        num_topics = request.args.get('num_topics', '5')
        num_tweets = request.args.get('num_tweets', '1000')  # default value
        topic_filter = request.args.get('topic')  # Get the topic filter from request

        endpoint = f'http://topic-socialabs.unikomcodelabs.id/topic?keyword={keyword}&num_topics={num_topics}&num_tweets={num_tweets}'
        response = requests.get(endpoint)
        response_data = response.json()

        data = response_data['data']['documents_topic']

        # data = clean_data(data)

        sentiment = Sentiment.classify_sentiment(data=data)

        # Calculate overall sentiment percentage
        sentiment_percentage = Sentiment.calculate_sentiment_percentages(data=sentiment)

        # Calculate sentiment percentage by topic
        sentiment_percentage_by_topic = Sentiment.calculate_sentiment_percentages_by_topic(data=sentiment)

        if topic_filter:
            # Filter tweets by topic
            filtered_sentiment = [item for item in sentiment if item['topic'] == topic_filter]
            return jsonify(
            {
                "sentiment": filtered_sentiment,
                "sentiment_percentage_by_topic": sentiment_percentage_by_topic[topic_filter]
            }), 200
        else:
            return jsonify(
            {
                "sentiment": sentiment,
                "sentiment_percentage": sentiment_percentage,
                "sentiment_percentage_by_topic": sentiment_percentage_by_topic
            }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

