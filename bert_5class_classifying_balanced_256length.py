import pandas as pd
import numpy as np
import torch
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)

#Results: 92.5% accuracy
#0.92 macro f1

# 1. Load and balance the dataset
df = pd.read_csv("selfdriving_5class_sentiment_v2.csv")
df = df.dropna(subset=["ArticleText", "Sentiment_5Class"])

# Upsample classes
classes = df['Sentiment_5Class'].unique()
dfs = [df[df['Sentiment_5Class'] == c] for c in classes]
max_size = max(len(d) for d in dfs)
resampled = [resample(d, replace=True, n_samples=max_size, random_state=42) for d in dfs]
df_balanced = pd.concat(resampled).sample(frac=1, random_state=42).reset_index(drop=True)

# 2. Encode labels
le = LabelEncoder()
df_balanced["label"] = le.fit_transform(df_balanced["Sentiment_5Class"])

# 3. Split into train and test
train_df, test_df = train_test_split(
    df_balanced, test_size=0.2, random_state=42, stratify=df_balanced["label"]
)

train_dataset = Dataset.from_pandas(train_df[["ArticleText", "label"]])
test_dataset = Dataset.from_pandas(test_df[["ArticleText", "label"]])

# 4. Tokenization with 256 max length
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["ArticleText"], truncation=True, padding=True, max_length=256)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# 5. Load the model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)

# 6. Training arguments
training_args = TrainingArguments(
    output_dir="./bert256-output",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_dir="./logs"
)

# 7. Metrics
def compute_metrics(pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

data_collator = DataCollatorWithPadding(tokenizer)

# 8. Set up Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# 9. Train the model
trainer.train()

# 10. Evaluate the model
predictions = trainer.predict(test_dataset)
y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=1)

# 11. Per-class performance
print("\nBERT (Balanced, 256 tokens) Per-Class Evaluation Report:\n")
print(classification_report(y_true, y_pred, target_names=le.classes_))