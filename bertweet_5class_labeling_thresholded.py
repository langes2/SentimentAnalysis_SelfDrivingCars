import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from collections import Counter

# 1. Load the dataset
df = pd.read_csv("selfdriving_5class_sentiment_v2.csv")
df = df.dropna(subset=["ArticleText"])

# 2. Load BERTweet model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
model = AutoModelForSequenceClassification.from_pretrained("vinai/bertweet-base", num_labels=3)
model.eval()  # Set model to evaluation mode

# Map 3 classes from BERTweet
id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}

# 3. Inference function
def classify_text(texts, batch_size=32):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encodings = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**encodings)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            max_probs, preds = torch.max(probs, dim=1)

            for p, pred in zip(max_probs, preds):
                results.append((p.item(), id2label[pred.item()]))
    return results

# 4. Apply classification
raw_predictions = classify_text(df["ArticleText"].tolist())

# 5. Threshold-based relabeling
final_labels = []
for prob, pred in raw_predictions:
    if prob < 0.50:
        final_labels.append("Neutral")
    elif pred == "Positive":
        if prob >= 0.75:
            final_labels.append("Positive")
        else:
            final_labels.append("Slightly Positive")
    elif pred == "Negative":
        if prob >= 0.78:
            final_labels.append("Negative")
        else:
            final_labels.append("Slightly Negative")
    else:  # original Neutral
        final_labels.append("Neutral")

# 6. Output new class distribution
final_class_distribution = Counter(final_labels)
print("\nNew Class Distribution with BERTweet + Thresholding:\n")
print(final_class_distribution)

# 7. Optional: Attach new labels back to DataFrame
df["RelabeledSentiment"] = final_labels

# 8. Optional: Save to CSV
# df.to_csv("selfdriving_5class_sentiment_bertweet_relabel.csv", index=False)