import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# -------------------------
# Image Transform
# -------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor()
])

# -------------------------
# Load Dataset
# -------------------------

dataset = datasets.ImageFolder(
    "../dataset",
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

print("Classes:", dataset.classes)
print("Total Images:", len(dataset))

# -------------------------
# Load Pretrained ResNet
# -------------------------

model = models.resnet50(pretrained=True)

# Change final layer for 2 classes
model.fc = nn.Linear(model.fc.in_features, 2)

# -------------------------
# Training Setup
# -------------------------

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 10

# -------------------------
# Training Loop
# -------------------------

for epoch in range(epochs):

    total_loss = 0

    for images, labels in loader:

        outputs = model(images)

        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print("Epoch:", epoch+1, "Loss:", total_loss)


# -------------------------
# Save Model
# -------------------------

torch.save(model.state_dict(),"deepfake_model.pth")

print("Training complete!")

torch.save(model.state_dict(), "model.pth")
print("Model saved!")