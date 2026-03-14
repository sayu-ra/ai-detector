import os
from PIL import Image

folder = "dataset"

for root, dirs, files in os.walk(folder):
    for file in files:
        if file.endswith(".jpeg"):
            path = os.path.join(root, file)

            img = Image.open(path)
            new_path = path.replace(".jpeg", ".jpg")

            img.save(new_path, "JPEG")
            os.remove(path)

print("Conversion complete")