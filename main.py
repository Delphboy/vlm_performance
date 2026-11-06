import json
import os
import sys

from PIL import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM
import urllib.request
import zipfile

def load_model(model_name):
    # Loads the model from Hugging Face according to the argument
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_name,
        device_map="auto"
    )

    print("Loaded processor and model")
    return processor, model

def caption_image(path, processor, model):
    # Queries a model to create a caption for an image at a given path
    image = Image.open(path)

    messages = [{
        "role": "user",
        "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Write one short sentence describing this image. Reply with only that sentence."},
        ]
    }]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    return caption

def load_dataset(split):
    # Download and unzip the full COCO dataset
    if not os.path.isdir("datasets/coco/img/val2014"):
        print("Downloading COCO dataset")
        os.makedirs("datasets/coco/img", exist_ok=True)

        urllib.request.urlretrieve(
            "http://images.cocodataset.org/zips/val2014.zip",
            "datasets/coco/img/val2014.zip"
        )

        with zipfile.ZipFile("datasets/coco/img/val2014.zip", 'r') as zip_ref:
            zip_ref.extractall("datasets/coco/img")

        os.remove("datasets/coco/img/val2014.zip")
    print("COCO dataset exists")

    # Download the Karpathy splits
    if not os.path.isfile("datasets/coco/dataset_coco.json"):
        print("Downloading Karpathy splits")
        os.makedirs("datasets/coco", exist_ok=True)
        urllib.request.urlretrieve(
            "https://github.com/Delphboy/karpathy-splits/raw/main/dataset_coco.json",
            "datasets/coco/dataset_coco.json"
        )
    print("Karpathy splits exist")

    # Loads each item in the split to an array and adds the local path
    with open(f"datasets/coco/{split}") as f:
        data = [item for item in json.load(f)["images"] if item["split"] == "test"]

    for item in data:
        destination = os.path.join('./datasets/coco/img', item["filepath"], item["filename"])
        item["local_path"] = destination

    return data

def main():
    # Loads the dataset, processor, and model, and initialises the results array
    model_name = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-4-E2B-it"
    split_name = sys.argv[2] if len(sys.argv) > 2 else "dataset_coco.json"

    data = load_dataset(split_name)
    processor, model = load_model(model_name)
    results = []

    # Creates results in the coco-captions format
    for item in data:
        caption = caption_image(item["local_path"], processor, model)
        results.append(
            {
                "image_id": item["cocoid"],
                "caption": caption
            }
        )

    # Outputs the results to a JSON file
    with open('results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("Captions stored in results.json")

if __name__ == "__main__":
    main()