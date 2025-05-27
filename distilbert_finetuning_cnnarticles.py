import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. Load CNN-labeled dataset
df = pd.read_csv("selfdriving_5class_sentiment_v2.csv")
df_cnn = df.tail(299).dropna(subset=["ArticleText", "Sentiment"])

# 2. Encode sentiment labels
le = LabelEncoder()
df_cnn["label"] = le.fit_transform(df_cnn["Sentiment"])

# 3. Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df_cnn[["ArticleText", "label"]])
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_ds = dataset["train"]
eval_ds = dataset["test"]

# 4. Tokenization
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["ArticleText"], truncation=True, padding=True, max_length=256)

train_ds = train_ds.map(tokenize, batched=True)
eval_ds = eval_ds.map(tokenize, batched=True)

# 5. Load model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

# 6. TrainingArguments
training_args = TrainingArguments(
    output_dir="./distilbert-cnn-3class",
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
    logging_dir="./logs"
)

# 7. Evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# 8. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

# 9. Train
trainer.train()

# 10. Save the model
trainer.save_model("./distilbert-cnn-3class")