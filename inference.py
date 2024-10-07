import os
import torch
import deepspeed
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def main():
    deepspeed.init_distributed()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    ds_config = "deepspeed_config.json"

    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
        model_parameters=None
    )

    questions = [
        "I absolutely love this movie, it's a masterpiece!",
        "The service at this restaurant was terrible and the food was bland.",
        "I'm feeling quite neutral about the whole situation.",
        "This new gadget is amazing, it has exceeded all my expectations!",
        "I'm really disappointed with the outcome of the game."
    ]

    labels = ["Negative", "Positive"]

    for input_text in questions:
        inputs = tokenizer(input_text, return_tensors="pt")
        inputs = {key: value.to(model_engine.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model_engine(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

        if rank == 0:
            print(f"Input: {input_text}")
            print(f"Prediction: {labels[predictions.item()]}")
        else:
            print(f"Worker {rank}: Inference completed for input: {input_text}")


if __name__ == "__main__":
        main()
