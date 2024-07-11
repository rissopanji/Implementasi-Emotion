from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import re
import nltk
import pickle
from nltk.tokenize import word_tokenize
from mpstemmer import MPStemmer
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')
stemmer = MPStemmer()

# Load the pre-trained emotion analysis model
model = load_model('./saved_model/model-bilstm.h5')

app = Flask(__name__)

class Emotion:
    def classify_emotion(data):

        def lower_case(text):
            return text.lower()

        def remove_tweet_special(text):
            text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
            text = text.encode('ascii', 'replace').decode('ascii')
            text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
            return text.replace("http://", " ").replace("https://", " ")

        def remove_number(text):
            return re.sub(r"\d+", "", text)

        def remove_punctuation(text):
            return text.translate(str.maketrans("", "", string.punctuation))

        def remove_whitespace_LT(text):
            return text.strip()

        def remove_whitespace_multiple(text):
            return re.sub('\s+',' ',text)

        def remove_singl_char(text):
            return re.sub(r"\b[a-zA-Z]\b", "", text)

        def remove_repeated_char(text):
            return re.sub(r'(.)\1+', r'\1', text)

        def word_tokenize_wrapper(text):
            return word_tokenize(text)

        normalizad_word = pd.read_csv("./utils/kamus-alay.csv")
        normalizad_word_dict = {}

        for index, row in normalizad_word.iterrows():
            if row[0] not in normalizad_word_dict:
                normalizad_word_dict[row[0]] = row[1] 

        def normalized_term(document):
            return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in document]

        def stem_wrapper(term):
            return [stemmer.stem(word) for word in term]

        stop_words = stopwords.words('indonesian')
        stop_words = [word for word in stop_words if word not in ['tidak', 'baik', 'jelek', 'jangan', 'belum', 'bukan', "enggak", "engga", "bener", "benar"]]
        stop_words.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                            'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                            'jd', 'jgn', 'sdh', 'aja', 'n', 't', 'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                            '&amp', 'yah'])

        txt_stopword = pd.read_csv("./utils/stopwords.txt", names=["stopwords"], header=None)
        stop_words.extend(txt_stopword["stopwords"][0].split(' '))
        stop_words = set(stop_words)

        def stopwords_removal(words):
            return [word for word in words if word not in stop_words]
        
        def replace_nan_with_none(data):
            return data.applymap(lambda x: None if pd.isna(x) else x)

        df = pd.DataFrame(data)

        if 'predicted_label' not in df.columns:
            df['predicted_label'] = np.nan
            df['probability_emotion'] = np.nan

        to_process_df = df[df['predicted_label'].isna()]

        if not to_process_df.empty:
            to_process_df['processed_text'] = to_process_df['full_text'].apply(lower_case)
            to_process_df['processed_text'] = to_process_df['processed_text'].apply(remove_tweet_special)
            to_process_df['processed_text'] = to_process_df['processed_text'].apply(remove_number)
            to_process_df['processed_text'] = to_process_df['processed_text'].apply(remove_punctuation)
            to_process_df['processed_text'] = to_process_df['processed_text'].apply(remove_whitespace_LT)
            to_process_df['processed_text'] = to_process_df['processed_text'].apply(remove_whitespace_multiple)
            to_process_df['processed_text'] = to_process_df['processed_text'].apply(remove_singl_char)
            to_process_df['processed_text'] = to_process_df['processed_text'].apply(remove_repeated_char)
            to_process_df['processed_text'] = to_process_df['processed_text'].apply(word_tokenize_wrapper)
            to_process_df['processed_text'] = to_process_df['processed_text'].apply(normalized_term)
            to_process_df['processed_text'] = to_process_df['processed_text'].apply(stem_wrapper)
            to_process_df['processed_text'] = to_process_df['processed_text'].apply(stopwords_removal)
            to_process_df['processed_text'] = to_process_df['processed_text'].apply(' '.join)

            print("Text preprocessing done!")

            with open('./utils/tokenizer-emotion.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)

            sequences = tokenizer.texts_to_sequences(to_process_df['processed_text'])
            padded_sequences = pad_sequences(sequences, maxlen=50, truncating='post', padding='post')

            predictions = model.predict(padded_sequences)
            
            emotion_labels = ['Neutral', 'Anger', 'Joy', 'Love', 'Sad', 'Fear']
            predicted_labels = []
            predicted_probabilities = []

            for pred in predictions:
                max_idx = np.argmax(pred)
                predicted_labels.append(emotion_labels[max_idx])
                predicted_probabilities.append({emotion_labels[i]: pred[i] for i in range(len(emotion_labels))})

            print("Prediction done!")

            to_process_df['predicted_label'] = predicted_labels
            to_process_df['probability_emotion'] = [predicted_probabilities[i][predicted_labels[i]] for i in range(len(predicted_labels))]

            df.update(to_process_df)

        df = replace_nan_with_none(df)

        return df.to_dict(orient='records')
    
    @staticmethod
    def calculate_emotion_percentages(data):
        total = len(data)
        emotion_counts = {'Neutral': 0, 'Anger': 0, 'Joy': 0, 'Love': 0, 'Sad': 0, 'Fear': 0}

        for item in data:
            emotion_counts[item['predicted_label']] += 1
        
        percentages = {emotion: (count / total) * 100 for emotion, count in emotion_counts.items()}
        return percentages

    @staticmethod
    def calculate_emotion_percentages_by_topic(data):
        topics = {}
        for item in data:
            topic = item.get('topic', 'unknown')
            if topic not in topics:
                topics[topic] = {'total': 0, 'Neutral': 0, 'Anger': 0, 'Joy': 0, 'Love': 0, 'Sad': 0, 'Fear': 0}
            topics[topic]['total'] += 1
            topics[topic][item['predicted_label']] += 1
        
        percentages_by_topic = {}
        for topic, counts in topics.items():
            percentages_by_topic[topic] = {emotion: (counts[emotion] / counts['total']) * 100 for emotion in counts if emotion != 'total'}
        
        return percentages_by_topic

@app.route('/classify_emotion', methods=['POST'])
def classify_emotion():
    data = request.json
    classified_data = Emotion.classify_emotion(data)
    return jsonify(classified_data)

@app.route('/emotion_percentages', methods=['POST'])
def emotion_percentages():
    data = request.json
    percentages = Emotion.calculate_emotion_percentages(data)
    return jsonify(percentages)

@app.route('/emotion_percentages_by_topic', methods=['POST'])
def emotion_percentages_by_topic():
    data = request.json
    percentages_by_topic = Emotion.calculate_emotion_percentages_by_topic(data)
    return jsonify(percentages_by_topic)

if __name__ == '__main__':
    app.run(debug=True)
