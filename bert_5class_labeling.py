import pandas as pd

# Load all necessary datasets
merged_data = pd.read_csv("/mnt/data/merged_selfdriving_sentiment_data.csv")
bert_data = pd.read_csv("/mnt/data/dl_labeled_unlabeled_data.csv")  # BERT (binary)
bertweet_data = pd.read_csv("/mnt/data/bertweet_labeled_unlabeled_data.csv")  # BERTweet (3-class)

# Clean up BERTweet labels
label_map = {
    'pos': 'Positive', 'Pos': 'Positive',
    'neg': 'Negative', 'Neg': 'Negative',
    'neu': 'Neutral', 'Neu': 'Neutral',
    'Neutral': 'Neutral'
}
bertweet_data['Sentiment_DL3'] = bertweet_data['Sentiment_DL3'].map(label_map)

# Use just URL as the join key (assuming it’s consistent across sets)
bertweet_map = bertweet_data.set_index('URL')['Sentiment_DL3']
bert_map = bert_data.set_index('URL')['Sentiment_DL']

# Add these labels to merged data
merged_data['bertweet_label'] = merged_data['URL'].map(bertweet_map)
merged_data['bert_label'] = merged_data['URL'].map(bert_map)

# Define the 5-class label generation logic
def get_5class_label(row):
    tfidf = row['Sentiment']
    bert = row['bert_label']
    bertweet = row['bertweet_label']
    
    if bertweet == 'Neutral':
        if bert == 'Positive':
            return 'Slightly Positive'
        elif bert == 'Negative':
            return 'Slightly Negative'
        elif tfidf == 'Neutral':
            return 'Neutral'
    return bertweet if bertweet in ['Positive', 'Negative', 'Neutral'] else None

# Apply function
merged_data['Sentiment_5Class'] = merged_data.apply(get_5class_label, axis=1)

# Output distribution
sentiment_5class_dist = merged_data['Sentiment_5Class'].value_counts()

# Save the final dataset
final_5class_path = "/mnt/data/selfdriving_5class_sentiment.csv"
merged_data.to_csv(final_5class_path, index=False)

sentiment_5class_dist, final_5class_path