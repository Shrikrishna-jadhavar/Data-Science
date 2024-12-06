"""@author: Shrikrishna Jadhavar
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import pickle
import os
import re
import pdfplumber
from PyPDF2 import PdfReader
import PyPDF2
import docx
import docx2txt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import pickle as pk
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer



st.title('RESUME CLASSIFICATION')
st.markdown('<style>h1{color: mediumvioletred;}</style>', unsafe_allow_html=True)
st.subheader('Welcome to Resume Classification App')

def getText(filename):
    fullText = '' # Create empty string 
    if filename.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx2txt.process(filename)
        fullText = doc  
    else:  
        with pdfplumber.open(filename) as pdf_file:
            pdoc = PyPDF2.PdfReader(filename)  # Use PdfReader instead of PdfFileReader
            number_of_pages = len(pdoc.pages)
            for page_num in range(number_of_pages):
                page = pdoc.pages[page_num]
                page_content = page.extract_text()
                fullText += page_content
    return fullText

def display(doc_file):
    if doc_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume = docx2txt.process(doc_file)
    else:
        with pdfplumber.open(doc_file) as pdf:
            resume = ''.join(page.extract_text() for page in pdf.pages)  
    return resume

def preprocess(sentence):
    sentence = str(sentence)
    sentence = sentence.lower()
    sentence = sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url = re.sub(r'http\S+', '', cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w) for w in filtered_words]
    return " ".join(lemma_words) 

file_type = pd.DataFrame([], columns=['Uploaded File', 'Predicted Profile'])
filename = []
predicted = []
skills = []

# Label mapping
label_mapping = {0: 'PeopleSoft', 1: 'SQL Developer', 2: 'React JS Developer', 3: 'Workday'}

# Example training data and model creation (replace with your actual model and data)
X_train = [
    "PeopleSoft",
    "SQL Developer",
    "React JS Developer",
    "Workday"
]
y_train = [0, 1, 2, 3]  # Labels: 0 = PeopleSoft, 1 = SQL Developer, 2 = React JS Developer, 3 = Workday

# Vectorize the training data.
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)

# Train the model.
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)

# Save the model and vectorizer together.
with open('naive_bayes_model.pkl', 'wb') as file:
    pickle.dump(nb, file)
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

# Load the model and vectorizer.
with open('naive_bayes_model.pkl', 'rb') as f:
    nb = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

upload_file = st.file_uploader('Upload Your Resumes', type=['doc', 'docx', 'pdf'], accept_multiple_files=True)
  
for doc_file in upload_file:
    if doc_file is not None:
        filename.append(doc_file.name)
        resume_text = display(doc_file)
        cleaned = preprocess(resume_text)
        input_vec = vectorizer.transform([cleaned])
        prediction = nb.predict(input_vec)[0]
        predicted.append(label_mapping[prediction])  # Map numerical label to profile name
        extText = getText(doc_file)
#        skills.append(extract_skills(extText))
        
if len(predicted) > 0:
    file_type['Uploaded File'] = filename
#    file_type['Skills'] = skills
    file_type['Predicted Profile'] = predicted
    st.table(file_type.style.format())
    
select_options = ['PeopleSoft', 'SQL Developer', 'React JS Developer', 'Workday']
st.subheader('Select as per Requirement')
option = st.selectbox('Fields', select_options)

if option == 'PeopleSoft':
    st.table(file_type[file_type['Predicted Profile'] == 'PeopleSoft'])
elif option == 'SQL Developer':
    st.table(file_type[file_type['Predicted Profile'] == 'SQL Developer'])
elif option == 'React JS Developer':
    st.table(file_type[file_type['Predicted Profile'] == 'React JS Developer'])
elif option == 'Workday':
    st.table(file_type[file_type['Predicted Profile'] == 'Workday'])
