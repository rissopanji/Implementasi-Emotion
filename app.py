import pickle
import os
import pandas as pd
import numpy as np
import re
import requests

from flask import Flask, request, jsonify
from model.tweet import Tweet
from service.emotion import Emotion
from service.crawler import TweetCrawler

auth_token = os.getenv("AUTH_TOKEN")

app = Flask(__name__)

def clean_data(data):
    for doc in data:
        for key, value in doc.items():
            if pd.isna(value):
                doc[key] = None
        if 'topic_probability':
            del doc['topic_probability']
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
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        if not (start_date and end_date):
            return jsonify({"error": "Start date and end date must be provided"}), 400
        
        keyword_regex = f".*{keyword}.*"
        cursor = Tweet.getTweetsByKeyword(keyword=keyword_regex, limit=jumlah_tweet, start_date=start_date, end_date=end_date)

        if not cursor or len(cursor) < jumlah_tweet:
            tweet_crawler = TweetCrawler(auth_token=auth_token, search_keyword=keyword, limit=jumlah_tweet, start_date=start_date, end_date=end_date)
            tweet_crawler.harvest_tweets()

            path_to_file = f"./tweets-data/{keyword}.csv"
            if os.path.exists(path_to_file):
                df = pd.read_csv(path_to_file)
                df.replace(np.nan, '', inplace=True, regex=True)

                data_crawling = []
                for index, row in df.iterrows():
                    tweet_data = row.to_dict()
                    data_crawling.append(tweet_data)
                
                emotion = Emotion.classify_emotion(data=data_crawling)
                Tweet.insertTweets(data_crawling)
                return jsonify(emotion), 200

            else:
                return jsonify({"error": "No tweets found"}), 404

        data = []
        for tweet in cursor:
            tweet_data = tweet.copy()
            tweet_data['_id'] = str(tweet['_id'])
            data.append(tweet_data)

        emotion = Emotion.classify_emotion(data=data)
        Tweet.updateEmotion(emotion)
        return jsonify(emotion), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500     

@app.route('/emotion-all', methods=['GET'])
def get_all_emotion():
    try:
        cursor = Tweet.getAllTweets()
        data = []
        for tweet in cursor:
            tweet_data = tweet.copy()
            tweet_data['_id'] = str(tweet['_id'])
            data.append(tweet_data)

        emotion = Emotion.classify_emotion(data=data)
        Tweet.updateEmotion(emotion)
        return jsonify(emotion), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500    

@app.route('/crawl', methods=['GET'])
def fetch_tweets():
    auth_token = request.args.get('auth_token')
    search_keyword = request.args.get('search_keyword')
    limit = request.args.get('limit', default=10, type=int)
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    tweet_crawler = TweetCrawler(auth_token=auth_token, search_keyword=search_keyword, limit=limit, start_date=start_date, end_date=end_date)
    tweet_crawler.harvest_tweets()
    return jsonify({"success" : "crawling data success!"}), 200

@app.route('/emotion', methods=['GET'])
def classify_emotion():
    try:
        keyword = request.args.get('keyword')
        num_topics = request.args.get('num_topics')
        num_tweets = request.args.get('num_tweets')
        topic_filter = request.args.get('topic')

        endpoint = f'http://topic-socialabs.unikomcodelabs.id/topic?keyword={keyword}&num_topics={num_topics}&num_tweets={num_tweets}'
        response = requests.get(endpoint)
        response_data = response.json()
        data = response_data['data']['documents_topic']

        emotion = Emotion.classify_emotion(data=data)
        emotion_percentage = Emotion.calculate_emotion_percentages(data=emotion)
        emotion_percentage_by_topic = Emotion.calculate_emotion_percentages_by_topic(data=emotion)

        if topic_filter:
            filtered_emotion = [item for item in emotion if item['topic'] == topic_filter]
            return jsonify({
                "emotion": filtered_emotion,
                "emotion_percentage_by_topic": emotion_percentage_by_topic[topic_filter]
            }), 200
        else:
            return jsonify({
                "emotion": emotion,
                "emotion_percentage": emotion_percentage,
                "emotion_percentage_by_topic": emotion_percentage_by_topic
            }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
