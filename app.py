import streamlit as st
import pickle
import string
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# --- 1. SET UP PAGE CONFIG ---
st.set_page_config(
    page_title="Spam Detector Pro",
    page_icon="🛡️",
    layout="centered"
)

# --- 2. SPEED HACK: NLTK SETUP ---
# Pre-downloading ensures the app doesn't stall on the first run
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')

download_nltk_data()
ps = PorterStemmer()

# --- 3. TEXT PREPROCESSING FUNCTION ---
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    # Remove special characters
    y = [i for i in text if i.isalnum()]
    
    # Remove stopwords and punctuation
    text = y[:]
    y = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    
    # Stemming (Porter Stemmer)
    text = y[:]
    y = [ps.stem(i) for i in text]
    
    return " ".join(y)

# --- 4. GENERIC PATH & MODEL LOADING ---
@st.cache_resource
def load_assets():
    # Get the absolute path to the directory containing app.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Generic paths to the files inside your 'model' folder
    vectorizer_path = os.path.join(current_dir, 'model', 'vectorizer.pkl')
    model_path = os.path.join(current_dir, 'model', 'model.pkl')
    
    # Load the pickle files
    try:
        tfidf = pickle.load(open(vectorizer_path, 'rb'))
        model = pickle.load(open(model_path, 'rb'))
        return tfidf, model
    except FileNotFoundError:
        st.error(f"Error: Model files not found in {os.path.join(current_dir, 'model')}. Please check your folder structure.")
        return None, None

tfidf, model = load_assets()

# --- 5. USER INTERFACE (UI) ---
st.title("🛡️ SMS/Email Spam Classifier")
st.write("This advanced detector uses a **Stacking Ensemble** model (SVC, Naive Bayes, Extra Trees) to identify spam with **98.07% Accuracy** and **100% Precision**.")

# Input area
input_sms = st.text_area("Enter the message you want to check:", height=150, placeholder="Paste your suspicious email or SMS here...")

# Sidebar info
st.sidebar.title("Model Insights")
st.sidebar.info("🚀 v2.0 Update: Improved detection for phishing URLs and complex scam patterns using Ensemble Stacking.")

st.sidebar.markdown("""
- **Precision:** 100% (No false alarms)
- **Accuracy:** 98.07%
- **Technique:** TF-IDF + Stacking Classifier
""")

# Prediction Logic
if st.button('Analyze Message'):
    if tfidf is not None and model is not None:
        if input_sms.strip() == "":
            st.warning("Please enter a message first!")
        else:
            # 1. Preprocess
            transformed_sms = transform_text(input_sms)
            # 2. Vectorize
            vector_input = tfidf.transform([transformed_sms])
            # 3. Predict
            result = model.predict(vector_input)[0]
            
            # 4. Display Results
            st.divider()
            if result == 1:
                st.error("### 🚨 Result: SPAM")
                st.write("Our model suggests this message has patterns commonly found in scam or promotional messages.")
            else:
                st.success("### ✅ Result: NOT SPAM")
                st.write("This message appears to be safe (Ham).")
    else:
        st.error("System error: Model assets failed to load.")

# --- 6. FOOTER ---
st.caption("Note: This tool is for educational purposes and provides predictions based on historical spam patterns.")