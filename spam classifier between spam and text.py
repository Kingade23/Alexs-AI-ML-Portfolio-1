import os
from pathlib import Path

# my spam classifier

# app.py
import streamlit as st # type: ignore
import joblib # pyright: ignore[reportMissingImports]
# Ensure relative paths to the .pkl files work when running via Streamlit
try:
    os.chdir(Path(__file__).parent)
except Exception:
    pass


# Load model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit UI
st.set_page_config(page_title="Spam Call Classifier", page_icon="📞", layout="centered")

st.title("📞 Spam Call Classifier")
st.write("Detect whether a call transcript or message is **Spam** or **Not Spam** using machine learning.")

# Text input
user_input = st.text_area("congratulation you have won 30 million naira, please confirm", height=150)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("⚠️ Dont respond.")
    else:
        # Transform and predict
        text_vec = vectorizer.transform([user_input])
        prediction = model.predict(text_vec)[0]
        probability = model.predict_proba(text_vec).max() * 100

        # Display result
        if prediction.lower() == "spam":
            st.error(f"🚨 SPAM DETECTED! (Confidence: {probability:.1f}%)")
        else:
            st.success(f"✅ NOT SPAM (Confidence: {probability:.1f}%)")

# Footer
st.markdown("---")
st.caption("Created by **Abdulaziz Alexander Abdulganiyu** | Powered by Streamlit & Scikit-learn")
