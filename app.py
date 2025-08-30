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

# NLTK Downloads - Ensure these are run before other NLTK imports
nltk_packages = ['omw-1.4', 'stopwords', 'punkt', 'wordnet']
for package in nltk_packages:
    try:
        nltk.data.find(f'tokenizers/{package}') # Updated path for punkt
    except LookupError:
        nltk.download(package)

from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import string
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter

# Removed the import for streamlit_wordcloud
# from streamlit_wordcloud import st_wordcloud # Or from streamlit_wordcloud_v2 import st_wordcloud

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
        # Ensure X is a Series for consistent behavior if a single string is passed
        if isinstance(X, str):
            X = pd.Series([X])
        return X.apply(lambda text: self.stemming(self.clean_text(text)))    

# Load the model, tfidf, and preprocessor
# It's good practice to use joblib with compression as you discussed, 
# assuming these files are saved as .joblib.gz
try:
    import joblib
    model = joblib.load('model.joblib.gz')
    tfidf = joblib.load('tfidf.joblib.gz')
    preprocessor = joblib.load('preprocessor.joblib.gz')
except FileNotFoundError:
    st.error("Model, TFIDF, or Preprocessor files not found. Please ensure 'model.joblib.gz', 'tfidf.joblib.gz', and 'preprocessor.joblib.gz' are in the same directory.")
    st.stop() # Stop the app if essential files are missing


# Load the data
try:
    data = pd.read_csv('prepared_data.csv')
except FileNotFoundError:
    st.error("Data file 'prepared_data.csv' not found. Please ensure it's in the same directory.")
    st.stop()

# MAKE THE APP
def main():
    # Preprocessor is loaded globally, no need to instantiate again here
    # preprocessor = Preprocessing() 
    
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Go to', ['Visualizations', 'Processing Steps', 'Prediction' ])

    if page == "Prediction":
        st.title("News Article Reliability Prediction")
        # User input
        user_input = st.text_area('Enter a news article:', height=200)

        # Predict
        if st.button('Predict'):
            if user_input:
                processed_input = preprocessor.transform(user_input) # Pass as string, Preprocessing handles Series conversion
                vectorized_input = tfidf.transform(processed_input)
                prediction = model.predict(vectorized_input)
                # Display the prediction
                st.write("---")
                st.subheader("Prediction Result:")
                if prediction == 0:
                    st.success('The news article is **reliable**.')
                else:
                    st.warning('The news article is **unreliable**.')
            else:
                st.info("Please enter some text to get a prediction.")

    elif page == "Processing Steps":
        st.title("Processing Steps for News Article Analysis")

        example_text = st.text_area('Enter a news article for step-by-step processing:', 
        """Chuck Todd: ’BuzzFeed Did Donald Trump a Political Favor’ - Breitbart,Jeff Poor,"Wednesday after   Donald Trump’s press conference at Trump Tower in New York City, NBC “Meet the Press” moderator Chuck Todd expressed his exasperation over the normalcy of what he called a “circus” surrounding Trump’s event.  “I was struck big picture wise which is of how normal a circus is now to us,” Todd said. “This was a circus. We’ve never seen a   a transition like we saw today where the press conference gets interrupted, you have a lawyer in here. The lawyer does half legal talk, half political spin. I’ve never seen that, using the lawyer to say he’s here to make America great again, and by the way I’m going to play constitutional lawyer. I don’t think this but clearly a constitutional lawyer told us we better not accept any of this money. So they made that exception. So I am struck at how normal crazy looked to us today. This was just a crazy scene, but this is the norm of Donald Trump. And in fact, this is where he’s most comfortable. And I will say this. just as a political show. if you’re Donald Trump, you want these press conferences because it made the press look disjointed, unorganized, all this stuff. And his people, you know, he just, it was a performance for his supporters and his people. ” Later in the segment, Todd decried what he saw as elements within the intelligence community being at odds with one another, then called a story put out by BuzzFeed a night earlier suggesting Trump had ties to Russia to be a “political favor. ” “Look, let’s be honest here,” Todd said. “Politically BuzzFeed did Donald Trump a political favor today by doing what they did by going ahead and making it all public because it allowed them to deny a specific without having to deal with the bigger picture. ” Follow Jeff Poor on Twitter @jeff_poor""", 
        height=300)

        if st.button('Show Processing Steps'):
            if example_text:
                st.subheader("Original Text:")
                st.code(example_text, language='text')

                st.subheader("1. Data Preprocessing")
                st.write("""The input text is first cleaned by converting it to lower case, removing digits, special characters, punctuations and extra spaces. It is then tokenized into individual words. Stop words are removed and the remaining words are lemmatized, and finally stemmed.""")
                processed_text = preprocessor.transform(example_text)[0] # Pass as string
                st.code(processed_text, language='text')

                st.subheader("2. Vectorization (TF-IDF)")
                st.write("The preprocessed text is then transformed into a numerical representation using TF-IDF vectorizer.")
                vectorized_text = tfidf.transform(pd.Series(processed_text))
                st.write("Shape of TF-IDF vector:", vectorized_text.shape)
                st.text("Sparse Matrix Representation (first 10 elements):")
                st.code(str(vectorized_text)[:200] + "...", language='text') # Show a snippet of the sparse matrix

                st.subheader("3. Prediction")
                st.write("The vectorized text is passed to the pre-trained model to predict if the news article is fake or real.")
                prediction = model.predict(vectorized_text)
                result = 'The news article is **reliable**.' if prediction == 0 else 'The news article is **unreliable**.'
                st.markdown(f"**Prediction**: {result}")
            else:
                st.info("Please enter some text to see the processing steps.")

    elif page == "Visualizations":
        st.title('Exploratory Data Analysis (EDA) Visualizations')

        # Labels Count Chart
        st.subheader("1. Labels Count Chart")
        labels_count = data['label'].value_counts()
        labels_count_bar = go.Figure(data=[go.Bar(x=['Reliable' if i == 0 else 'Unreliable' for i in labels_count.index], y=labels_count.values)], 
                                     layout=go.Layout(title='Distribution of News Article Labels', height=400, width=800,
                                                      xaxis_title="Label", yaxis_title="Count"))
        st.plotly_chart(labels_count_bar, use_container_width=True)

        # Top 20 Authors (assuming 'author' column is in the original df, not 'prepared_data.csv')
        # If 'prepared_data.csv' only has 'processed_content' and 'label', you'll need the original df for author.
        # For now, I'll assume you have access to the original 'df' if you want to plot this.
        # If not, you'd need to modify 'prepared_data.csv' to include 'author' or skip this plot.
        try:
            original_df = pd.read_csv('train.csv') # Load original data if 'author' is not in prepared_data.csv
            st.subheader("2. Top 20 Authors (from original dataset)")
            # Assuming you filled NaNs for author in original df for processing
            original_df['author'].fillna('Unknown', inplace=True)
            top_authors = original_df['author'].value_counts().head(20)
            top_authors_bar = go.Figure(data=[go.Bar(x=top_authors.index, y=top_authors.values)],
                                         layout=go.Layout(title='Top 20 Authors by Article Count', height=400, width=800,
                                                          xaxis_title="Author", yaxis_title="Number of Articles"))
            st.plotly_chart(top_authors_bar, use_container_width=True)
        except FileNotFoundError:
            st.warning("Original 'train.csv' not found. Skipping 'Top 20 Authors' visualization.")
        except KeyError:
            st.warning("The 'author' column is not available in the loaded data. Skipping 'Top 20 Authors' visualization.")


        # Word Frequency
        st.subheader("3. Word Frequency (Top 25 Most Common Words)")
        
        # Calculate word frequency from the processed content
        all_words = ' '.join(data['processed_content']).split()
        counter = Counter(all_words)
        most_common = counter.most_common(25) # Display top 25 words
        x_freq, y_freq = zip(*most_common)
        fig_freq = go.Figure(data=[go.Bar(x=list(x_freq), y=list(y_freq))])
        fig_freq.update_layout(title='Top 25 Most Common Words in Processed Content', height=450, width=800,
                               xaxis_title="Word", yaxis_title="Frequency")
        st.plotly_chart(fig_freq, use_container_width=True)

        # Removed the Interactive Word Cloud section
        # st.subheader("4. Interactive Word Cloud")
        # st.write("This word cloud visualizes the most frequent words in the processed content.")
        
        # # Prepare data for st_wordcloud - it expects a list of dictionaries
        # word_freq_list = [{'text': word, 'value': count} for word, count in counter.items()]
        
        # # Display the word cloud using the component
        # st_wordcloud(word_freq_list, 
        #              width=900, height=500, # Adjust size as needed
        #              # Optional: customize colors, font, etc.
        #              # cloud_color='random-light', background_color='white',
        #              # font_size_max=80, font_size_min=10
        #             )


if __name__ == '__main__':
    main()