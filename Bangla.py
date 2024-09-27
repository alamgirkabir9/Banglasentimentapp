import streamlit as st
import pandas as pd
from Banglanlpdeeplearn.model import model_train
from Banglanlpdeeplearn.text_process import preprocess_text
from Banglanlpdeeplearn.predict import predict_sentiment

# Title of the app
st.title("Bangla Sentiment Analysis :heart_eyes: :cry:")

@st.cache_resource
def load_model(file_path):
    df = pd.read_csv(file_path)
    df['processed_text'] = df['text'].apply(preprocess_text)
    model1, model2, tokenizer, encoder, X_test, y_test, max_length = model_train(df, 'processed_text', 'label')
    return model1, model2, tokenizer, encoder, X_test, y_test, max_length

# Load dataset and model
file_path = "https://raw.githubusercontent.com/alamgirkabir9/Banglasentimentapp/refs/heads/main/bangla.csv"
model1, model2, tokenizer, encoder, X_test, y_test, max_length = load_model(file_path)

# Display dataset preview
st.write("## Dataset Preview")
df = pd.read_csv(file_path)  # Load dataset for preview
st.dataframe(df, height=200)

# Evaluate the model
loss, accuracy = model1.evaluate(X_test, y_test)
st.metric(label="Accuracy", value=f"{accuracy:.2f}")
st.progress(accuracy)

# User input for prediction
user_input = st.text_area("Enter Bangla text for prediction", "")

if st.button("Show Prediction"):
    if user_input:
        predicted_label = predict_sentiment(user_input, model1, tokenizer, encoder, max_length)
        st.success(f"**Predicted Sentiment**: {predicted_label}")
    else:
        st.warning("Please enter text for prediction.")

st.markdown(
    """
    <hr style="border: 1px solid #e74c3c;">
    <div style="text-align:center;">
        <p style="color:yellow;font-weight:bold;">Bangla Sentiment Analysis App | Powered by Alamgir</p>
    </div>
    """, 
    unsafe_allow_html=True
)
