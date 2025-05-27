import pandas as pd
from transformers import pipeline

# Load your merged dataset
df = pd.read_csv("merged_selfdriving_sentiment_data.csv")

# Filter out rows that still don't have a sentiment
unlabeled = df[df['Sentiment'].isna()].copy()
unlabeled = unlabeled.dropna(subset=['ArticleText'])

# Truncate to 512 characters (max token input for most BERT-based models)
unlabeled['short_text'] = unlabeled['ArticleText'].str[:512]

# Load the 3-class sentiment analysis model
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="finiteautomata/bertweet-base-sentiment-analysis"
)

# Predict in batches
batch_size = 32
results = []
for i in range(0, len(unlabeled), batch_size):
    batch = unlabeled['short_text'].iloc[i:i+batch_size].tolist()
    try:
        preds = sentiment_pipeline(batch)
        results.extend(preds)
    except:
        results.extend([{"label": "neutral", "score": 0.0}] * len(batch))

# Assign results
unlabeled['Sentiment_DL3'] = [res['label'].capitalize() for res in results]

# Save the new dataset
unlabeled.to_csv("bertweet_labeled_unlabeled_data.csv", index=False)

# Optional: Check distribution
print(unlabeled['Sentiment_DL3'].value_counts())