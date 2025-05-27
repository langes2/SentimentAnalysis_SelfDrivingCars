import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

#Results: 90.87% accuracy!

# Load dataset
df = pd.read_csv("selfdriving_5class_sentiment_v2.csv")
df = df.dropna(subset=['ProcessedText', 'Sentiment_5Class'])

# Balance the classes by upsampling
classes = df['Sentiment_5Class'].unique()
dfs = [df[df['Sentiment_5Class'] == c] for c in classes]
max_size = max(len(d) for d in dfs)
resampled = [resample(d, replace=True, n_samples=max_size, random_state=42) for d in dfs]
df_balanced = pd.concat(resampled).sample(frac=1, random_state=42).reset_index(drop=True)

# Tokenization
vocab_size = 10000
max_len = 200
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(df_balanced['ProcessedText'])
sequences = tokenizer.texts_to_sequences(df_balanced['ProcessedText'])
padded = pad_sequences(sequences, maxlen=max_len)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df_balanced['Sentiment_5Class'])
y_cat = to_categorical(y, num_classes=5)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    padded, y_cat, test_size=0.2, random_state=42, stratify=y_cat)

# Build BiLSTM model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=100, input_length=max_len),
    Bidirectional(LSTM(128)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(5, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping
early_stop = EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop]
)

# Final test evaluation
loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy with Class Balancing + BiLSTM: {acc:.4f}")

# 🔍 Per-class performance
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

label_names = le.classes_
report = classification_report(y_true, y_pred, target_names=label_names, digits=4)
print("\nPer-Class Evaluation Report:\n")
print(report)

# Accuracy & loss plots
plt.figure(figsize=(10, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_names, yticklabels=label_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()