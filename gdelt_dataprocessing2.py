import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download resources only once
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load datasets
gdelt = pd.read_csv("gdelt_selfdriving_cleaned.csv")
cnn = pd.read_csv("Reformatted_SelfDrivingNews.csv")

# Standardize CNN columns
cnn = cnn.rename(columns={
    'Link': 'URL',
    'Web': 'Source',
    'Article': 'ArticleText',
    'Label': 'Sentiment'
})

# Preprocess text
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = nltk.word_tokenize(str(text).lower())
    tokens = [re.sub(r'[^a-z]', '', token) for token in tokens]
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and token.isalpha()]
    return ' '.join(tokens)

cnn['ProcessedText'] = cnn['ArticleText'].apply(preprocess_text)

# Add placeholder for GDELT sentiment
gdelt['Sentiment'] = None

# Combine datasets
combined = pd.concat([gdelt, cnn[gdelt.columns]], ignore_index=True)

# Save it
combined.to_csv("merged_selfdriving_sentiment_data.csv", index=False)
print("Merged dataset saved as 'merged_selfdriving_sentiment_data.csv'")