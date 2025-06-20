import streamlit as st
from PIL import Image
from predictor import predict_image

st.set_page_config(page_title="Animal Species Prediction", layout="centered")
st.title("🐾 Animal Species Prediction")

st.write("Upload an image of an animal and the model will predict its species.")

uploaded_file = st.file_uploader("Upload Animal Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Classifying..."):
        label, confidence = predict_image(image)
    
    st.success(f"**Prediction:** {label}")
    st.info(f"**Confidence:** {confidence:.2f}%")
