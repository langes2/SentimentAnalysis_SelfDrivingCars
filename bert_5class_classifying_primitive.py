import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from datasets import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from transformers.trainer_callback import EarlyStoppingCallback

#Results: 73% accuracy
# 0.41 macro f1

# 1. Load and prepare data
df = pd.read_csv("selfdriving_5class_sentiment_v2.csv")
df = df.dropna(subset=["ArticleText", "Sentiment_5Class"])

# Encode labels
le = LabelEncoder()
df["label"] = le.fit_transform(df["Sentiment_5Class"])

# Train/test split
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

# Convert to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df[["ArticleText", "label"]])
test_dataset = Dataset.from_pandas(test_df[["ArticleText", "label"]])

# 2. Load tokenizer and tokenize data
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["ArticleText"], truncation=True, padding=True, max_length=512)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# 3. Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)

# 4. Set up Trainer
training_args = TrainingArguments(
    output_dir="./bert-output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

def compute_metrics(pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

data_collator = DataCollatorWithPadding(tokenizer)

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

# 5. Train model
trainer.train()

# 6. Evaluate model
predictions = trainer.predict(test_dataset)
y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=1)

# 7. Per-class evaluation
print("\nBERT Per-Class Evaluation Report:\n")
print(classification_report(y_true, y_pred, target_names=le.classes_))