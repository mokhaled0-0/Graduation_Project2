import re
from textblob import TextBlob
from nltk.stem.wordnet import WordNetLemmatizer
import itertools
import numpy as np
import nltk
import joblib
import preprocessor as p
from .data_collection_and_preprocessing import clean_tweets
import warnings
import concurrent.futures
from .Sentiment import preprocess
import pandas as pd
import pickle
import joblib


warnings.filterwarnings('ignore')


class sentiment_analysis_code():

    def predicts(self, text):
        file_path = 'C:/Users/ALRYADA/Desktop/sent model/Sentiment-Lm.pkl'
        file_path1 = 'C:/Users/ALRYADA/Desktop/sent model/vectoriser-ngran.pkl'
        model = joblib.load(file_path)
        vectoriser = joblib.load(file_path1)
        # Predict the sentiment
        textdata = vectoriser.transform(preprocess(text))
        sentiment = model.predict(textdata)
        prop = model.predict_proba(textdata)
        # Predict the sentiment
        probabilty_negative = prop[:, 0]
        probabilty_positive = prop[:, 1]
        # Make a list of text with sentiment.
        data = []
        for text, pred, negative, positive in zip(text, sentiment, probabilty_negative, probabilty_positive):
            data.append((text, pred, negative, positive))

        # Convert the list into a Pandas DataFrame.
        df = pd.DataFrame(
            data, columns=['text', 'sentiment', 'negative', 'positive'])
        df = df.replace([0, 1], ["negative", "positive"])
        print(df.iloc[0]['sentiment'], df.iloc[0]['text'],
              probabilty_negative[0], probabilty_positive[0])
        return df.iloc[0]['sentiment'], df.iloc[0]['text'], probabilty_negative[0], probabilty_positive[0]
