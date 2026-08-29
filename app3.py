import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Chest X-Ray Pneumonia Detection",
    layout="centered"
)

st.title(" Chest X-Ray Pneumonia Detection")
st.write("Upload a chest X-ray image to detect whether it indicates **NORMAL** or **PNEUMONIA**.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("chest_xray_model.keras")

try:
    model = load_model()
except Exception as e:
    st.error("Error loading the model. Please ensure 'chest_xray_model.keras' is in the same directory.")

uploaded_file = st.file_uploader("Choose a Chest X-Ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    st.image(image, caption="Uploaded X-Ray Image", use_container_width=True)
    
    if st.button("Analyze X-Ray"):
        with st.spinner("Analyzing image..."):
            img_resized = image.resize((128, 128))
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)
            
            prediction = model.predict(img_batch)[0][0]
            
            st.divider()
            
            if prediction > 0.5:
                confidence = prediction * 100
                st.error(f"###  Result: PNEUMONIA Detected")
                st.write(f"**Confidence:** {confidence:.2f}%")
            else:
                confidence = (1 - prediction) * 100
                st.success(f"### Result: NORMAL (No Pneumonia)")
                st.write(f"**Confidence:** {confidence:.2f}%")