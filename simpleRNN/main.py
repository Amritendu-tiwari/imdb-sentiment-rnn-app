import numpy as np 
import tensorflow as tf 
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model



word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

## load the pre-trained model with Relu activation 
# model = load_model('simple_rnn_imdb.h5
model = load_model('./simple_rnn_imdb.h5')


#helper fucntion 
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3, '?')for i in encoded_review])


## Function to perprocess user input
# def preprocess_text(text):
#     words = text.lower().split()
#     encoded_review = [word_index.get(word, 2) + 3 for word in words]
#     padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
#     return padded_review

num_words = 10000
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}


def preprocess_text(text):
    words = text.lower().split()
    encoded_review = []
    for word in words:
        index = word_index.get(word)
        if index is not None and index < num_words:
            encoded_review.append(index + 3)  # +3 for reserved tokens
        else:
            encoded_review.append(2)  # 2 is <UNK> token
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review



### step 3: prediction function 
# def predict_sentiment(review):
#     preprocess_input = preprocess_text(review)
    
#     predicition=model.predict(preprocess_input)
    
#     sentiment = 'positive' if predicition [0] [0] > 0.5 else 'Negative'
    
#     return sentiment, predicition[0][0]


## streamlit.app

import streamlit as st 
st.title('IMDB Movie Review sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')


user_input = st.text_area('Movie Review')

if st.button('Classify'):
    
    preprocess_input=preprocess_text(user_input)
    
    ## Make prediction
    prediction=model.predict(preprocess_input)
    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    
    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Pls enter a movie review.')
