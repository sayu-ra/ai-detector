import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

class Detector:
    def __init__(self, model_dir):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
        # Use the absolute path for weights
        self.model.load_weights(f"{model_dir}/tf_model.h5")

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="tf", truncation=True, padding=True)
        logits = self.model(inputs).logits
        probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
        return probs[1] * 100