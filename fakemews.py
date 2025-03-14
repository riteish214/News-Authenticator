import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Set Page Configuration (MUST be the first Streamlit command)
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# File paths
MODEL_FILE = 'fakenews_model.pkl'
VECTORIZER_FILE = 'vectorizer.pkl'

# Sample dataset to train if model is not found
news_data = {
    'text': [
        'Aliens landed in New York.',
        'Government launches new policy to reduce pollution.',
        'Man claims to have time-traveled 1000 years ahead.',
        'Vaccines approved by WHO for new virus strain.',
        'Fake doctor caught practicing without a license.',
        'Global warming is just a hoax, says man from future.',
        'Scientists find new planet similar to Earth.',
        'New energy drink promises to increase life expectancy.'
    ],
    'label': [0, 1, 0, 1, 0, 0, 1, 1]
}

def train_model():
    df = pd.DataFrame(news_data)
    X = df['text']
    y = df['label']

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X = vectorizer.fit_transform(X)

    model = LogisticRegression()
    model.fit(X, y)

    # Save model and vectorizer
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)

    st.success("🎉 Model successfully trained and saved!")

# Force train model if not found
def load_or_train_model():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
        st.warning("🛠 Model or Vectorizer not found. Training now...")
        train_model()
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
    return model, vectorizer

# Load the model and vectorizer
model, vectorizer = load_or_train_model()

# Streamlit App UI
st.title('📰 Fake News Detection Web App')

# Text Input
news_text = st.text_area("📝 Enter News Text:")

if st.button('🔍 Check News'):
    if news_text.strip():
        # Predict News
        text_vectorized = vectorizer.transform([news_text])
        prediction = model.predict(text_vectorized)

        # Show Results
        if prediction[0] == 1:
            st.success("✅ This News is Real!")
        else:
            st.error("❌ This News is Fake!")
    else:
        st.warning("⚠️ Please enter some text to check.")

# File Upload
st.write("---")
st.subheader("📄 Upload a Text File to Check Multiple News")
uploaded_file = st.file_uploader("Upload a .txt file", type=['txt'])

if uploaded_file is not None:
    # Read File Content
    file_text = uploaded_file.read().decode("utf-8")
    file_lines = file_text.split('\n')

    # Analyze Each Line
    results = []
    for line in file_lines:
        if line.strip():
            text_vectorized = vectorizer.transform([line])
            prediction = model.predict(text_vectorized)
            result = "Real News" if prediction[0] == 1 else "Fake News"
            results.append(f"{line} --> {result}")

    # Display Results
    st.write("### File Results:")
    for res in results:
        st.write(res)

    # Download Results
    result_text = '\n'.join(results)
    st.download_button(
        label="📥 Download Results",
        data=result_text,
        file_name="fakenews_results.txt",
        mime="text/plain"
    )

# Hosting Instructions
st.write("---")
