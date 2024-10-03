import streamlit as st
import requests
from PIL import Image
import io

st.title("Image Detection Web App")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Button to send the image to the API
    if st.button('Detect'):
        # Convert the image to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_bytes = img_bytes.getvalue()

        # Send the image to the API
        # response = requests.post("http://api:8000/predict/", files={"file": img_bytes})
        response = requests.post("http://fastapi:8000/predict/", files={"file": img_bytes})
        if response.status_code == 200:
            result_image = Image.open(io.BytesIO(response.content))
            st.image(result_image, caption="Image with Detection Boxes", use_column_width=True)
        else:
            print(response)
            st.error("Error in API call")
