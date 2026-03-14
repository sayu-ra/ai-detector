import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def run_training(data_path, save_dir):

    print("Loading dataset...")

    df = pd.read_csv(data_path)

    print("Dataset loaded:", len(df), "rows")

    texts = df["text"].astype(str).tolist()
    labels = df["generated"].astype(int).tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=128)

    train_dataset = TensorDataset(
        torch.tensor(train_enc["input_ids"]),
        torch.tensor(train_enc["attention_mask"]),
        torch.tensor(train_labels)
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )

    optimizer = AdamW(model.parameters(), lr=5e-5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    epochs = 3

    print("Training started...")

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for batch in train_loader:

            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{epochs} Loss: {avg_loss}")

    os.makedirs(save_dir, exist_ok=True)

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print("Training complete.")
    print("Model saved to:", save_dir)


if __name__ == "__main__":

    data_path = "./dataset/ai_generated_essays.csv"
    save_dir = "./model_assets/text_detector"

    run_training(data_path, save_dir)