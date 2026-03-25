import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont

# Load environment variables
HF_TOKEN = st.secrets["HF_TOKEN"]

API_URL = "https://router.huggingface.co/hf-inference/models/facebook/detr-resnet-101"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}


def query_image(image_bytes, content_type="image/jpeg"):
    response = requests.post(
        API_URL,
        headers={"Content-Type": content_type, **headers},
        data=image_bytes
    )
    return response.json()


st.title("DETR Object Detection Viewer")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    uploaded_file.seek(0)
    img_bytes = uploaded_file.read()
    content_type = "image/png" if uploaded_file.type == "image/png" else "image/jpeg"

    with st.spinner("Detecting objects..."):
        output = query_image(img_bytes, content_type)

    if isinstance(output, list):
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

        st.write("Detected Objects:")
        for obj in output:
            label = obj["label"]
            score = obj["score"]
            box = obj["box"]

            x_min = box["xmin"]
            y_min = box["ymin"]
            x_max = box["xmax"]
            y_max = box["ymax"]

            # Draw rectangle + label
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
            draw.text((x_min, y_min - 15), f"{label} {score:.2f}", fill="red", font=font)

            st.write(f"- {label}: {score:.2f}")

        st.image(image, caption="Detected Objects", use_container_width=True)
    else:
        st.write("No objects detected or API error:")
        st.write(output)

