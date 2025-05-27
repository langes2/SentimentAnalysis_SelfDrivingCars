import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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

#poor performance
#Results: 54% accuracy

# 1. Load data
df = pd.read_csv("selfdriving_5class_sentiment_v2.csv")
df = df.dropna(subset=["ArticleText", "Sentiment_5Class"])

# 2. Encode labels
le = LabelEncoder()
df["label"] = le.fit_transform(df["Sentiment_5Class"])

# 3. Split into train/test BEFORE oversampling
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)

# 4. Upsample training set only
train_dfs = [train_df[train_df['label'] == i] for i in range(len(le.classes_))]
max_count = max(len(d) for d in train_dfs)
resampled_train = [resample(d, replace=True, n_samples=max_count, random_state=42) for d in train_dfs]
train_df_balanced = pd.concat(resampled_train).sample(frac=1, random_state=42)

# 5. Convert to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df_balanced[["ArticleText", "label"]])
test_dataset = Dataset.from_pandas(test_df[["ArticleText", "label"]])

# 6. Tokenization (max length 512)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["ArticleText"], truncation=True, padding=True, max_length=512)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# 7. Load model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)

# 8. Training configuration
training_args = TrainingArguments(
    output_dir="./bert512-output",
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

def compute_metrics(pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# 9. Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# 10. Train
trainer.train()

# 11. Final evaluation
predictions = trainer.predict(test_dataset)
y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=1)

print("\nFinal Evaluation Report (DistilBERT 512 tokens):\n")
print(classification_report(y_true, y_pred, target_names=le.classes_))