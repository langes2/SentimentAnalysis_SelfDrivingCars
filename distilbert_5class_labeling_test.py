import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from collections import Counter
from tqdm import tqdm

# 1. Load fine-tuned DistilBERT model (binary)
model_path = "./distilbert-cnn-3class"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

# 2. Load your full dataset
df = pd.read_csv("selfdriving_5class_sentiment_v2.csv")
df = df.dropna(subset=["ArticleText"])
texts = df["ArticleText"].tolist()

# 3. Run predictions in batches
final_labels = []
batch_size = 32

with torch.no_grad():
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        encodings = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        outputs = model(**encodings)
        probs = softmax(outputs.logits, dim=-1)
        max_probs, preds = torch.max(probs, dim=1)

        for prob, pred in zip(max_probs, preds):
            p = prob.item()
            label = pred.item()  # 0 = Negative, 1 = Positive

            if p < 0.50:
                final_labels.append("Neutral")
            elif label == 1:
                final_labels.append("Positive" if p >= 0.75 else "Slightly Positive")
            elif label == 0:
                final_labels.append("Negative" if p >= 0.78 else "Slightly Negative")
            else:
                final_labels.append("Neutral")

# 4. Display final class distribution
distribution = Counter(final_labels)
print("\nFinal Class Distribution (5-Class via Thresholding):\n")
for label, count in distribution.items():
    print(f"{label}: {count}")

# 5. (Optional) Save predictions to CSV
# df["RelabeledSentiment"] = final_labels
# df.to_csv("distilbert_5class_thresholded_labels.csv", index=False)