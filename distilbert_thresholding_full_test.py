import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from tqdm import tqdm

# 1. Load model and tokenizer
model_path = "./distilbert-cnn-3class"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

# 2. Load full dataset
df = pd.read_csv("selfdriving_5class_sentiment_v2.csv")
df = df.dropna(subset=["ArticleText"])
texts = df["ArticleText"].tolist()

# 3. Predict in batches
batch_size = 32
predictions = []

with torch.no_grad():
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        encodings = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        outputs = model(**encodings)
        probs = softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=-1).tolist()
        predictions.extend(preds)

# 4. Map class IDs back to labels (based on training order)
id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}
predicted_labels = [id2label[p] for p in predictions]

# 5. Display class distribution
from collections import Counter
distribution = Counter(predicted_labels)
print("\nClass Distribution (DistilBERT, No Thresholding):\n")
print(distribution)