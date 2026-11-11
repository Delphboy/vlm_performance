import json
import os
import urllib.request

# Load the COCO Karpathy dataset information
with open("datasets/coco/dataset_coco.json", "r") as file:
    data = json.load(file)

# Select the first five test images
test_images = [
    image for image in data["images"]
    if image["split"] == "test"
][:5]

# Folder where images will be saved
folder = "datasets/coco/img/val2014"
os.makedirs(folder, exist_ok=True)

# Download images that are not already available
for image in test_images:
    filename = image["filename"]
    path = os.path.join(folder, filename)

    if os.path.isfile(path):
        print("Already exists:", filename)
        continue

    url = (
        "https://s3.amazonaws.com/images.cocodataset.org/"
        "val2014/" + filename
    )

    print("Downloading:", filename)

    try:
        urllib.request.urlretrieve(url, path)
        print("Downloaded successfully:", filename)
    except Exception as error:
        print("Download failed:", filename, error)

print("Download process completed!")