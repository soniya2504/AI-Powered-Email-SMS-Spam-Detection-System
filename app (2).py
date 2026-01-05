import streamlit as st
import joblib
import pandas as pd
from PIL import Image

# Load model
model = joblib.load("svm_spam_model.pkl")

# Load cleaned data
df = pd.read_csv("cleaned_spam_mail.csv")

# UI Config
st.set_page_config(page_title="AI Spam Detector", layout="wide")

# Header
st.title("AI Powered Email & SMS Spam Detector")

# Load image banner
img = Image.open("A_banner_image_for_an_AI_Spam_Detector_application.png.png")
st.image(img, use_container_width=True)


# Input box
st.subheader("Enter a message or email to check:")
user_input = st.text_area("", height=120)

# Buttons section
col1, col2, col3, col4 = st.columns(4)

with col1:
    predict_btn = st.button("Predict Spam")
with col2:
    metrics_btn = st.button("Show Model Metrics")
with col3:
    misclassified_btn = st.button("Show Misclassified Samples")
with col4:
    download_btn = st.button("Download Cleaned Data")

# Output
if predict_btn:
    if user_input.strip():
        result = model.predict([user_input])[0]
        if result == "spam":
            st.error("This is SPAM")
        else:
            st.success("This is NOT SPAM")
    else:
        st.warning("Please enter some text!")

# Metrics display
if metrics_btn:
    st.subheader("Dataset Label Distribution:")
    st.bar_chart(df['label'].value_counts())

    st.subheader("Sample Cleaned Data:")
    st.dataframe(df[['clean_email','label']].head(10))

# Misclassified samples
if misclassified_btn:
    st.subheader("Examples where model was wrong:")
    wrong = df[df['label'] != model.predict(df['clean_email'].tolist())]
    st.dataframe(wrong[['clean_email','label']].head(15))

# Download cleaned data
if download_btn:
    st.download_button(
        "Click to Download CSV",
        data=df.to_csv(index=False),
        file_name="cleaned_spam_mail.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("Built by Soniya Murugesan | NLP, ML, Power BI, Python, SQL")
