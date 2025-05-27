import pandas as pd
from transformers import pipeline

# Load your merged dataset
df = pd.read_csv("merged_selfdriving_sentiment_data.csv")

# Select unlabeled entries
unlabeled = df[df['Sentiment'].isna()].copy()
unlabeled = unlabeled.dropna(subset=['ArticleText'])
unlabeled['short_text'] = unlabeled['ArticleText'].str[:512]

# Load sentiment pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Run in batches
batch_size = 32
results = []
for i in range(0, len(unlabeled), batch_size):
    batch = unlabeled['short_text'].iloc[i:i+batch_size].tolist()
    preds = sentiment_pipeline(batch)
    results.extend(preds)

# Assign results
label_map = {"POSITIVE": "Positive", "NEGATIVE": "Negative"}
unlabeled['Sentiment_DL'] = [label_map.get(r['label'].upper(), "Neutral") for r in results]

# Save to CSV
unlabeled.to_csv("dl_labeled_unlabeled_data.csv", index=False)

# Show class distribution
print(unlabeled['Sentiment_DL'].value_counts())