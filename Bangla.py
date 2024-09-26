import streamlit as st
import pandas as pd
from Banglanlpdeeplearn.model import model_train
from Banglanlpdeeplearn.text_process import preprocess_text
from Banglanlpdeeplearn.predict import predict_sentiment

st.title("Bangla Sentiment Analysis :heart_eyes: :cry:")

file_path = "E:\\NLP\\bangla_sentiment_data.csv"
df = pd.read_csv(file_path)
print(df.head())
df['processed_text'] = df['text'].apply(preprocess_text)
model1, model2, tokenizer, encoder, X_test, y_test, max_length = model_train(df, 'processed_text', 'label')
st.write("## Dataset Preview")
st.dataframe(df, height=200)

loss, accuracy = model1.evaluate(X_test, y_test)
st.metric(label="Accuracy", value=f"{accuracy:.2f}")
st.progress(accuracy)
st.write("## Predict Sentiment")
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