import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


# -------------------------
# CNN Model (same as training)
# -------------------------

class FakeDetector(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(64*28*28,128),
            nn.ReLU(),
            nn.Linear(128,2)
        )

    def forward(self,x):

        x = self.conv(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


# -------------------------
# Load trained model
# -------------------------

model = FakeDetector()

model.load_state_dict(torch.load("models/deepfake_model.pth"))

model.eval()


# -------------------------
# Image transform
# -------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor()
])


# -------------------------
# Load test image
# -------------------------

image = Image.open("test_images/test.jpeg")

image = transform(image)

image = image.unsqueeze(0)


# -------------------------
# Prediction
# -------------------------

with torch.no_grad():

    output = model(image)

    prediction = torch.argmax(output,1).item()


classes = ["fake","real"]

print("Prediction:", classes[prediction])