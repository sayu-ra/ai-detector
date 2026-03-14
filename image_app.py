import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from model import DeepfakeCNN

model = DeepfakeCNN()
model.load_state_dict(torch.load("deepfake_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

st.title("Deepfake Image Detector")

uploaded = st.file_uploader("Upload Image")

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img)

    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)

    pred = torch.argmax(output)

    if pred == 0:
        st.error("FAKE IMAGE")
    else:
        st.success("REAL IMAGE")