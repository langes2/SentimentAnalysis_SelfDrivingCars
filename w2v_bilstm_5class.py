import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#Results: 58.72% accuracy

# Load your dataset
df = pd.read_csv("selfdriving_5class_sentiment_v2.csv")
df = df.dropna(subset=['ProcessedText', 'Sentiment_5Class'])

# Tokenize
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['ProcessedText'])
sequences = tokenizer.texts_to_sequences(df['ProcessedText'])
padded = pad_sequences(sequences, maxlen=200)

# Encode the labels
le = LabelEncoder()
y = le.fit_transform(df['Sentiment_5Class'])  # Outputs 0 to 4
y_cat = to_categorical(y, num_classes=5)

# Split into training/testing
X_train, X_test, y_train, y_test = train_test_split(
    padded, y_cat, test_size=0.2, random_state=42, stratify=y_cat)

# Build BiLSTM model
model = Sequential([
    Embedding(input_dim=10000, output_dim=100, input_length=200),
    Bidirectional(LSTM(128)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(5, activation='softmax')  # 5-class output
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=7, batch_size=32, validation_split=0.2)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")