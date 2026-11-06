import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# Load SmolVLM2
model_name = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

processor = AutoProcessor.from_pretrained(model_name)

model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    dtype=torch.float32
).to("cpu")

model.eval()

# Load the Karpathy test images
with open("datasets/coco/dataset_coco.json", "r") as file:
    data = json.load(file)

test_images = [
    image for image in data["images"]
    if image["split"] == "test"
][:5]

results = []

# Generate one caption for each image
for item in test_images:
    filename = item["filename"]
    image_path = "datasets/coco/img/val2014/" + filename

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image in one sentence."}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False
        )

    generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]

    caption = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0].strip()

    print(filename)
    print("Generated caption:", caption)
    print()

    results.append({
        "filename": filename,
        "generated_caption": caption,
        "reference_captions": [
            sentence["raw"] for sentence in item["sentences"]
        ]
    })

# Save the generated captions for evaluation
with open("results.json", "w") as file:
    json.dump(results, file, indent=4)

print("All five captions generated and saved!")