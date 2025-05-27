import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
data = pd.read_csv("/mnt/data/gdelt_selfdriving_master.csv")

# Keep only essential columns
data = data[['Date', 'URL', 'Source', 'Title', 'ArticleText']]

# Remove duplicates based on article content
data = data.drop_duplicates(subset=['ArticleText'])

# Expanded keyword set
keywords = [
    "driverless", "tesla", "self", "driving", "drive", "autonomous",
    "robot", "robotic", "auto", "automated", "self-driving", "accident", "accidents"
]

# Filter based on keyword presence in the article text (case insensitive)
pattern = '|'.join(keywords)
filtered_data = data[data['ArticleText'].str.contains(pattern, case=False, na=False)]

# Drop any rows with missing article text
filtered_data = filtered_data.dropna(subset=['ArticleText'])

# Initialize NLP preprocessing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenization and lowercase
    tokens = nltk.word_tokenize(text.lower())

    # Remove non-alphabetic characters
    tokens = [re.sub(r'[^a-z\s]', '', token) for token in tokens]

    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and token.isalpha()]

    return ' '.join(tokens)

# Apply preprocessing
filtered_data['ProcessedText'] = filtered_data['ArticleText'].apply(preprocess_text)

# Save the cleaned data
filtered_data.to_csv("/mnt/data/cleaned_selfdriving_data.csv", index=False)

print(f"Filtered and processed dataset size: {filtered_data.shape}")