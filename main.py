import json
import os
from PIL import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM
import urllib.request

def load_model():
    model_name = "google/gemma-4-E2B-it"

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_name,
        device_map="auto"
    )

    return processor, model

def caption_image(path, processor, model):
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

def load_dataset():
    if not os.path.isfile("dataset_coco.json"):
        urllib.request.urlretrieve(
            "https://github.com/Delphboy/karpathy-splits/raw/main/dataset_coco.json",
            "dataset_coco.json"
        )

    with open("dataset_coco.json") as f:
        data = [item for item in json.load(f)["images"] if item["split"] == "test"]

    for item in data:
        destination = os.path.join('./images', item["filepath"], item["filename"])
        item["local_path"] = destination
        if os.path.isfile(destination):
            continue

        os.makedirs(os.path.dirname(destination), exist_ok=True)
        urllib.request.urlretrieve(
            f"http://images.cocodataset.org/{item['filepath']}/{item['filename']}",
            destination
        )

    return data

def main():
    data = load_dataset()
    processor, model = load_model()
    results = []

    for item in data:
        caption = caption_image(item["local_path"], processor, model)
        results.append(
            {
                "image_id": item["cocoid"],
                "caption": caption
            }
        )

    with open('results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()