# Reload the full dataset and reapply the DistilBERT binary model with 5-class thresholding logic

import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from tqdm import tqdm
from collections import Counter

# Load full dataset
df = pd.read_csv("/mnt/data/selfdriving_5class_sentiment_v2.csv")
df = df.dropna(subset=["ArticleText"])
texts = df["ArticleText"].tolist()

# Load binary DistilBERT model
model_path = "./distilbert-cnn-3class"  # path to your saved 2-class model
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

# Inference with 5-class thresholding
batch_size = 32
final_labels = []

with torch.no_grad():
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        encodings = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        outputs = model(**encodings)
        probs = softmax(outputs.logits, dim=-1)
        max_probs, preds = torch.max(probs, dim=1)

        for prob, pred in zip(max_probs, preds):
            p = prob.item()
            label = pred.item()
            if p < 0.50:
                final_labels.append("Neutral")
            elif label == 1:  # Positive
                final_labels.append("Positive" if p >= 0.75 else "Slightly Positive")
            elif label == 0:  # Negative
                final_labels.append("Negative" if p >= 0.78 else "Slightly Negative")
            else:
                final_labels.append("Neutral")

# Count final label distribution
thresholded_binary_distilbert_distribution = Counter(final_labels)
thresholded_binary_distilbert_distribution