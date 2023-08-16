import streamlit as st
import pickle
import pandas as pd
import docx2txt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the trained SVM classifier
loaded_clf = pickle.load(open('clf.pkl', 'rb'))

# Define a function to preprocess the input resume text
def preprocess_resume(text):
     # Convert text to lowercase
    text = text.lower()

    # Remove any non-alphabetic characters except spaces
    text = re.sub('[^a-zA-Z ]', '', text)

    # Tokenize the text
    words = nltk.word_tokenize(text)

    # Lemmatize words to their base form
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Reconstruct the text from processed words
    processed_text = ' '.join(words)

    return processed_text

# Define the Streamlit app
def main(cv):  # Pass the CountVectorizer as an argument
    st.title('Resume Classification App')
    st.write("Upload your DOCX file below and click the 'Predict' button to get the category.")

    # User input for uploading the DOCX file
    uploaded_file = st.file_uploader("Upload your DOCX file", type=['docx'])

    if st.button('Predict'):
        if uploaded_file is not None:
            # Extract text from the uploaded DOCX file
            resume_text = docx2txt.process(uploaded_file)

            # Preprocess the extracted text
            preprocessed_text = preprocess_resume(resume_text)

            # Use the loaded CountVectorizer from training for transforming the test data
            x_test = cv.transform([preprocessed_text])

            # Use the loaded classifier to make predictions
            prediction = loaded_clf.predict(x_test)

            # Map numerical label back to category name
            category_map = {0: 'React', 1: 'SQL', 2: 'PeopleSoft', 3: 'workday r'}
            predicted_category = category_map[prediction[0]]

            # Display the predicted category
            st.success(f"Predicted Category: {predicted_category}")

if __name__ == "__main__":
    # Load the trained SVM classifier
    loaded_clf = pickle.load(open('clf.pkl', 'rb'))

    # Load the CountVectorizer used during training
    cv = pickle.load(open('cv.pkl', 'rb'))

    # Call the main function and pass the CountVectorizer as an argument
    main(cv)