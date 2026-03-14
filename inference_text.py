import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class Detector:

    def __init__(self, model_dir=None):

        # Load model from HuggingFace instead of local folder
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=2
        )

        self.model.eval()

    def predict(self, text):

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        return probs[0][1].item() * 100
