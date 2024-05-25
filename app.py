import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

nltk_packages = ['omw-1.4', 'stopwords', 'punkt', 'wordnet']

for package in nltk_packages:
    try:
        nltk.data.find(package)
    except LookupError:
        nltk.download(package)

import re
import string
from wordcloud import WordCloud
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter

st.set_page_config(layout='wide', initial_sidebar_state='expanded')

class Preprocessing(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.port_stem = PorterStemmer()

    def fit(self, X, y=None):
        return self
    
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)        
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.strip()
        text = word_tokenize(text)
        text = [word for word in text if word not in self.stop_words]
        text = [self.lemmatizer.lemmatize(word) for word in text]
        return ' '.join(text)
    
    def stemming(self, text):
        words = text.split()
        stemmed_words = [self.port_stem.stem(word) for word in words]
        return ' '.join(stemmed_words)

    def transform(self, X, y=None):
        return X.apply(lambda text: self.stemming(self.clean_text(text)))    

# load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# load the tfidf
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# load the pipeline
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)
        
# Load the data
data = pd.read_csv('prepared_data.csv')

def main():
    preprocessor = Preprocessing()
    
    # User input
    user_input = st.text_area('Enter a news article:')

    # Predict
    if st.button('Predict'):
        user_input = preprocessor.transform(pd.Series(user_input))
        user_input = tfidf.transform(user_input)
        prediction = model.predict(user_input)
        # Display the prediction
        if prediction == 0:
            st.write('The news article is reliable.')
        else:
            st.write('The news article is unreliable.')

if __name__ == '__main__':
    main()