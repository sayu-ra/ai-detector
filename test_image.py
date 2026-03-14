import torch
from PIL import Image
from torchvision import transforms
from model import DeepfakeCNN

# load model
model = DeepfakeCNN()
model.load_state_dict(torch.load("deepfake_model.pth"))
model.eval()

# image transform (VERY IMPORTANT)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# load image
img = Image.open("test_images/test5.jpg").convert("RGB")

# apply transform
img = transform(img)

# add batch dimension
img = img.unsqueeze(0)

# prediction
with torch.no_grad():
    output = model(img)

prediction = torch.argmax(output)

if prediction == 0:
    print("Prediction: AI-generated image")
else:
    print("Prediction: REAL image")