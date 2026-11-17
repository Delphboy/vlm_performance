import json
import os
import sys

from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM
import urllib.request
import zipfile
import argparse

def load_model(model_name):
    # Loads the model from Hugging Face according to the argument
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_name,
        device_map = {"": 0 if torch.cuda.is_available() else "cpu"},
    )

    print("Loaded processor and model on device:", model.device)
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

def load_dataset(split, dataset_directory):
    # Download and unzip the full COCO dataset
    if not os.path.isdir(os.path.join(dataset_directory, "img/val2014")):
        print("Downloading COCO dataset")
        os.makedirs(os.path.join(dataset_directory, "img"), exist_ok=True)

        urllib.request.urlretrieve(
            "http://images.cocodataset.org/zips/val2014.zip",
            os.path.join(dataset_directory, "img/val2014.zip")
        )

        with zipfile.ZipFile(os.path.join(dataset_directory, "img/val2014.zip"), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(dataset_directory, "img"))

        os.remove(os.path.join(dataset_directory, "img/val2014.zip"))
    print("COCO dataset exists")

    # Download the Karpathy splits
    if not os.path.isfile(os.path.join(dataset_directory, "dataset_coco.json")):
        print("Downloading Karpathy splits")
        os.makedirs(dataset_directory, exist_ok=True)
        urllib.request.urlretrieve(
            "https://github.com/Delphboy/karpathy-splits/raw/main/dataset_coco.json",
            os.path.join(dataset_directory, "dataset_coco.json")
        )
    print("Karpathy splits exist")

    # Loads each item in the split to an array and adds the local path
    with open(os.path.join(dataset_directory, split)) as f:
        data = [item for item in json.load(f)["images"] if item["split"] == "test"]

    for item in data:
        destination = os.path.join(dataset_directory, "img", item["filepath"], item["filename"])
        item["local_path"] = destination

    return [data[0]]

def main():
    # Processes arguments
    parser = argparse.ArgumentParser(prog="vlm-performance")
    parser.add_argument(
        "-m",
        "--model",
        help="Model name from HuggingFace (defaults to google/gemma-4-E2B-it)",
        default="google/gemma-4-E2B-it"
    )
    parser.add_argument(
        "-s",
        "--split",
        help="Split file name (defaults to dataset_coco.json)",
        default="dataset_coco.json"
    )
    parser.add_argument(
        "-p",
        "--path",
        help="Path of the dataset and split files (defaults to datasets/coco)",
        default="datasets/coco"
    )
    args = parser.parse_args()
    model_name = args.model
    split_name = args.split
    dataset_directory = args.path

    # Loads the dataset, processor, and model, and initialises the results array
    data = load_dataset(split_name, dataset_directory)
    processor, model = load_model(model_name)
    results = []

    # Creates results in the coco-captions format
    print("Creating captions...")
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