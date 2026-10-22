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

def caption_image(path):
    image = Image.open(path)

    processor, model = load_model()

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
    print(caption_image(sys.argv[1]))

if __name__ == "__main__":
    main()