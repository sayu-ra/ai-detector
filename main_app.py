import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

# TEXT MODEL
from inference_text import Detector

# IMAGE MODEL
from image_model import DeepfakeCNN

# -----------------------------
# Load Text Model
# -----------------------------
text_detector = Detector("./model_assets_text/text_detector")

# -----------------------------
# Load Image Model
# -----------------------------
image_model = DeepfakeCNN()
image_model.load_state_dict(torch.load("deepfake_model.pth", map_location="cpu"))
image_model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# -----------------------------
# WEBSITE UI
# -----------------------------

st.title("AI Content Detector")

option = st.sidebar.selectbox(
    "Choose Detection Type",
    ["Text Detection","Image Detection"]
)

# -----------------------------
# TEXT DETECTION
# -----------------------------

if option == "Text Detection":

    st.header("AI Text Detector")

    user_text = st.text_area("Paste text here")

    if st.button("Analyze Text"):

        if user_text.strip() == "":
            st.warning("Please enter text")
        else:

            score = text_detector.predict(user_text)

            st.write("AI Probability:", round(score,2), "%")

            if score > 50:
                st.error("Likely AI Generated")
            else:
                st.success("Likely Human")

# -----------------------------
# IMAGE DETECTION
# -----------------------------

if option == "Image Detection":

    st.header("Deepfake Image Detector")

    uploaded = st.file_uploader("Upload Image")

    if uploaded:

        img = Image.open(uploaded).convert("RGB")

        st.image(img)

        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = image_model(img)

        pred = torch.argmax(output)

        if pred == 0:
            st.error("Fake Image")
        else:
            st.success("Real Image")