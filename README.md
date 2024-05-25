# Fake News Detection Project

## Overview

This project is designed to build a machine learning model to detect fake news. It preprocesses the text data, builds and evaluates several models, and finally deploys the best model using a Streamlit app for user interaction.

###Try the [Streamlit app](https://ziadmostafa1-fake-news-detection-app-mvav1l.streamlit.app/)!

## Table of Contents

- [Fake News Detection Project](#fake-news-detection-project)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Project Structure](#project-structure)
  - [Usage](#usage)
    - [Data Preprocessing](#data-preprocessing)
    - [Model Training](#model-training)
    - [Model Evaluation](#model-evaluation)
    - [Model Deployment](#model-deployment)
  - [Conclusion](#conclusion)

## Installation

To set up this project, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/ZiadMostafa1/Fake_News_Detection.git
   cd Fake_News_Detection
   ```

2. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

4. Download necessary NLTK data:
   ```sh
   python -m nltk.downloader stopwords punkt wordnet omw-1.4
   ```

## Project Structure

```
fake-news-detection/
├── train.csv                # Training data
├── prepared_data.csv        # Processed data for the Streamlit app
├── model.pkl                # Trained model
├── tfidf.pkl                # TFIDF vectorizer
├── preprocessor.pkl         # Preprocessing pipeline
├── app.py                   # Streamlit app for deployment
├── requirements.txt         # Dependencies
├── README.md                # Project documentation
└── project_notebook.ipynb   # Main notebook for data processing and model training
```

## Usage

### Data Preprocessing

The data preprocessing steps involve cleaning the text data, removing stop words, and applying lemmatization and stemming. These steps are implemented in the `Preprocessing` class.

```python
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
```

### Model Training

The script `notebook.py` performs the following steps:
1. Loads and preprocesses the data.
2. Splits the data into training and testing sets.
3. Trains a Logistic Regression and a Naive Bayes classifier.
4. Evaluates the models using cross-validation and classification reports.
5. Saves the best model, preprocessing pipeline, and TFIDF vectorizer using `pickle`.

### Model Evaluation

The models are evaluated based on their cross-validation scores and classification reports. The Logistic Regression and Naive Bayes classifiers are compared to identify the best performing model.

### Model Deployment

The `app.py` file contains a Streamlit app for deploying the model. It allows users to input a news article and get a prediction on whether it is reliable or not.

```python
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
```

To run the Streamlit app, use the following command:
```sh
streamlit run app.py
```

## Conclusion

This project demonstrates a complete pipeline for fake news detection, from data preprocessing and model training to model evaluation and deployment. The Streamlit app allows for user interaction with the trained model, making it easy to classify news articles as reliable or unreliable.
