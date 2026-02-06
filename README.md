**🛡️ Stop-Spam-Now: Advanced SMS/Email Classifier**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://stop-spam-now.streamlit.app/)

Stop-Spam-Now is a high-precision machine learning web application designed to identify fraudulent messages. By combining multiple specialized algorithms, the app achieves an elite balance of accuracy and reliability, ensuring that "Ham" (legitimate) messages are never misclassified as "Spam."

**💎 The Precision Advantage**

In the world of spam detection, a False Positive (marking a real message as spam) is far more damaging than a False Negative.

  **Precision: 100%** — Guaranteed zero false alarms.
  
  **Accuracy: 98.07%** — Top-tier detection rate across diverse scam patterns.
  
  **Model Type**: Ensemble Stacking (SVC + Naive Bayes + Extra Trees).

**🛠️ Technical Stack**

  **Frontend**: Streamlit
  
  **Language**: Python 3.9+
  
  **NLP Library**: NLTK (Tokenization, Stopword removal, Porter Stemming)
  
  **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
  
  **Machine Learning**: Scikit-Learn

**🧠 Model Architecture**

This project utilizes a Stacking Classifier. Unlike a single model, stacking uses a "Meta-Classifier" to aggregate the predictions of multiple base models:
  1.Multinomial Naive Bayes: Excellent for frequency-based word patterns.
  2.Support Vector Classifier (SVC): Powerful at finding high-dimensional boundaries between classes.
  3.Extra Trees: An ensemble of decision trees that reduces variance.
  4.Final Estimator: A Logistic Regression model that makes the final "hard" decision based on the consensus of the base models.

### 📂 Project Structure

```text
.
├── model/
│   ├── model_new.pkl
│   └── vectorizer_new.pkl
├── app.py
├── requirements.txt
└── README.md   
```


**💻 Local Setup**

**1 .Clone the Repo:**

  git clone https://github.com/Akansha051991/sms-spam-classification.git
  
**2.Install Requirements:**

  pip install -r requirements.txt
  
**3.Run App:**

  streamlit run app.py
