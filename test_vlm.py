from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

model_name = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    torch_dtype=torch.float32
).to("cpu")

print("Model loaded successfully!")

from PIL import Image

# Load our first COCO test image
image_path = "datasets/coco/img/val2014/COCO_val2014_000000391895.jpg"
image = Image.open(image_path).convert("RGB")

print("COCO image loaded successfully!")

# Prepare the image and captioning instruction
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

# Generate a caption using the CPU
print("Generating caption...")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)

caption = processor.batch_decode(
    outputs[:, inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
)[0]

print("Generated caption:", caption)