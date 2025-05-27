import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#Results: 62.06% accuracy with early stopping

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

from tensorflow.keras.callbacks import EarlyStopping

# Early stopping callback
early_stop = EarlyStopping(
    monitor='val_accuracy',     # or 'val_loss'
    patience=2,                 # stops after 2 stagnant epochs
    restore_best_weights=True  # bring back the best model
)

# Train with early stopping
history = model.fit(
    X_train, y_train,
    epochs=20,                # you can use a higher upper bound now
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate the best model on test data
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")