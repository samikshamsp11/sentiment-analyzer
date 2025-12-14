import streamlit as st
from model import model

st.title("AI-Powered Customer Feedback Sentiment Analysis Tool")

st.write("Enter customer feedback below and click Analyze.")

user_input = st.text_area("Customer Feedback")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        prediction = model.predict([user_input])
        st.success(f"Predicted Sentiment: {prediction[0]}")
