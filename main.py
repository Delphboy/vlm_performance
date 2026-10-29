import json
import os
import sys
from PIL import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM

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

def main():
    processor, model = load_model()
    results = []

    for file in os.listdir('./images'):
        caption = caption_image(os.path.join('./images',file), processor, model)
        results.append(
            {
                "image_id": file,
                "caption": caption
            }
        )

    with open('results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()