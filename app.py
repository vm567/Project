import streamlit as st
from transformers import pipeline

# Load the sentiment analysis model from Hugging Face
classifier = pipeline('sentiment-analysis')

# Create a Streamlit app
st.title('Sentiment Analysis with Hugging Face')
st.write('Enter some text and we will predict its sentiment!')

# Add a text input box for the user to enter text
text_input = st.text_input('Enter text here')

# When the user submits text, run the sentiment analysis model on it
if st.button('Submit'):
    # Predict the sentiment of the text using the Hugging Face model
    sentiment = classifier(text_input)[0]['label']
    
    # Display the sentiment prediction to the user
    st.write(f'Sentiment: {sentiment}')