
from config.db import db
from bson import ObjectId
from pymongo import results  
from datetime import datetime
from flask import jsonify

class Tweet:
    
    def getAllTweets():
        return db.all_final_2class.find({}, {'full_text': 1})
    

    def getTweetsByKeyword(keyword, limit, start_date, end_date):
        start_datetime = datetime.strptime(f"{start_date} 00:00:00 +0000", "%Y-%m-%d %H:%M:%S %z")
        end_datetime = datetime.strptime(f"{end_date} 23:59:59 +0000", "%Y-%m-%d %H:%M:%S %z")
        print(f"Filtering tweets from {start_datetime} to {end_datetime}")  # Debugging
        
        # Contoh query dengan aggregation pipeline
        cursor = db.all_final_2class.aggregate([
        {
            '$match': {
                'full_text': {'$regex': keyword, '$options': 'i'}
            }
        },
        {
            '$addFields': {
                'parsed_date': {'$toDate': '$created_at'}
            }
        },
        {
            '$match': {
                'parsed_date': {'$gte': start_datetime, '$lte': end_datetime}
            }
        },
        {
            '$limit': limit
        }
        ])
        return list(cursor)
    
    def insertTweets(data):
        try:
            result = db.all_final_2class.insert_many(data)
            print(f"Inserted {len(result.inserted_ids)} documents")
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def updateSentiment(data):

        for item in data:
            # Convert the string representation of ObjectId back to ObjectId
            document_id = ObjectId(item['_id'])

            update_data = {
                "predicted_label": item['predicted_label'],
                "probability": item['probability'],
                # "processed_text": item['processed_text'],
            }

            result = db.all_final_2class.update_one({'_id': document_id}, {'$set': update_data})
            print(f"Updating {document_id}: {result.matched_count} document(s) matched, {result.modified_count} document(s) updated.")

            