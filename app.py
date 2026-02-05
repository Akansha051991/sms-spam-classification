import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Download NLTK data inside the app
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# 1. Get the absolute path to the folder where app.py is located
current_dir = os.path.dirname(__file__)

# 2. Point to the 'models' folder specifically
# Note the 's' at the end of 'models' to match your folder name
vectorizer_path = os.path.join(current_dir, 'models', 'vectorizer.pkl')
model_path = os.path.join(current_dir, 'models', 'model.pkl')

# 3. Load the pickles using these verified paths
@st.cache_resource
def load_models():
    tfidf = pickle.load(open(vectorizer_path, 'rb'))
    model = pickle.load(open(model_path, 'rb'))
    return tfidf, model
tfidf, model = load_models()

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    
# 4. Display with Colors and Effects
    if result == 1:
        st.error("🚨 **SPAM DETECTED!**")
        st.snow() 
    else:
        st.success("✅ **NOT SPAM**")
        st.balloons()