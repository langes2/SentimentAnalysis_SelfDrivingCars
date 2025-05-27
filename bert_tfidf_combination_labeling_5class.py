# Redefine 5-class assignment logic with TF-IDF Neutral priority
def get_5class_label_prioritize_tfidf(row):
    tfidf = row['Sentiment']
    bert = row['bert_label']
    bertweet = row['bertweet_label']
    
    # Priority: if both TF-IDF and BERTweet agree on Neutral
    if tfidf == 'Neutral' and bertweet == 'Neutral':
        return 'Neutral'
    
    # Next: BERTweet neutral + BERT sentiment for slightly positive/negative
    if bertweet == 'Neutral':
        if bert == 'Positive':
            return 'Slightly Positive'
        elif bert == 'Negative':
            return 'Slightly Negative'
    
    # Fallback to BERTweet label
    return bertweet if bertweet in ['Positive', 'Negative', 'Neutral'] else None

# Apply new logic
merged_data['Sentiment_5Class'] = merged_data.apply(get_5class_label_prioritize_tfidf, axis=1)

# Save and show updated distribution
final_5class_path_v2 = "/mnt/data/selfdriving_5class_sentiment_v2.csv"
merged_data.to_csv(final_5class_path_v2, index=False)
sentiment_5class_dist_v2 = merged_data['Sentiment_5Class'].value_counts()

sentiment_5class_dist_v2, final_5class_path_v2