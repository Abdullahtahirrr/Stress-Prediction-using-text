import streamlit as st

import joblib
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('wordnet')
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Load the model outside the main function
voting_clf = joblib.load('voting_clf.pkl')

tf = joblib.load('tfidf_vectorizer.pkl')


# Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('wordnet')

# Instantiate your stemmer
ps = PorterStemmer()
# Define your stop words set
stop_words = set(stopwords.words('english'))

def clean(text):
    try:
        # Convert text to lowercase
        text = str(text).lower()

        # Remove brackets
        text = re.sub('[][)(]', ' ', text)

        # Remove URLs
        text = [word for word in text.split() if not urlparse(word).scheme]
        text = ' '.join(text)

        # Remove escape characters
        text = re.sub(r'\@\w+', '', text)

        # Remove HTML tags 
        text = re.sub(re.compile("<.*?>"), '', text)

        # Remove non-alphanumeric characters
        text = re.sub("[^A-Za-z0-9]", ' ', text)  

        # Tokenize text
        tokens = word_tokenize(text)

        # Remove stopwords
        tokens = [word for word in tokens if word not in stop_words]

        # Lemmatize or stem tokens (choose one approach)
        tokens = [ps.stem(word) for word in tokens]  # For stemming
        # tokens = [lemmatizer.lemmatize(word) for word in tokens]  # For lemmatization

        # Join tokens back into text
        text = ' '.join(tokens)

        return text
    except Exception as ex:
        print(text, "\n")
        print("Error ", ex)

def preprocess_input(text):
    processed_text = clean(text) 
    vectorized_input = tf.transform([processed_text]) 
    return vectorized_input

def predict_class(input_text):
# Preprocess the input
    preprocessed_input = preprocess_input(input_text)
# Predict the class
    predicted_class = voting_clf.predict(preprocessed_input)
    return predicted_class[0] 

   
   
def main():
    st.title('Stress Classifier')
    st.write('Hello Abdullah, How are you feeling today? Express your feelings below!')
    # Input box for user to enter a sentence
    user_input = st.text_input('Enter a sentence:')
    
    # Button to trigger the prediction
    if st.button('Predict'):
        if user_input:
            # Predict the class
            predicted_class = predict_class(user_input)
            
            if predicted_class.lower() == "stress":  # Ensure case-insensitivity
                st.warning('Predicted class: Stress. Take some rest and relax!')
            else:
                st.success('You seem stress-free! Enjoy your day.')
        else:
            st.warning('Please enter a sentence.')

if __name__ == '__main__':
    main()
