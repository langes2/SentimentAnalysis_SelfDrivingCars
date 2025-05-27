import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    AutoConfig
)
import torch.nn as nn
from sklearn.metrics import classification_report

# 1. Load and filter dataset (excluding neutral)
df = pd.read_csv("selfdriving_5class_sentiment_v2.csv")
df = df[df["Sentiment_5Class"] != "Neutral"]  # filter out missing class
df = df.dropna(subset=["ArticleText", "Sentiment_5Class"])

# Encode labels to 4 classes
le = LabelEncoder()
df["label"] = le.fit_transform(df["Sentiment_5Class"])

# Train/test split
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)

# Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df[["ArticleText", "label"]])
test_dataset = Dataset.from_pandas(test_df[["ArticleText", "label"]])

# 2. Tokenization
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["ArticleText"], truncation=True, padding=True, max_length=256)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# 3. Compute class weights for 4 classes
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df["label"]),
    y=train_df["label"]
)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# 4. Define model with class-weighted loss
from transformers import BertForSequenceClassification

class BertWithWeightedLoss(BertForSequenceClassification):
    def __init__(self, config, class_weights):
        super().__init__(config)
        self.class_weights = class_weights

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            loss = loss_fct(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

# Load pretrained weights
config = AutoConfig.from_pretrained("bert-base-uncased", num_labels=4)
base_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", config=config)
model = BertWithWeightedLoss(config, class_weights)
model.load_state_dict(base_model.state_dict(), strict=False)

# 5. TrainingArguments
training_args = TrainingArguments(
    output_dir="./bert-4class-weighted",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_dir="./logs"
)

# 6. Evaluation metrics
def compute_metrics(pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# 7. Trainer setup
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

# 8. Train
trainer.train()

# 9. Evaluate
predictions = trainer.predict(test_dataset)
y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=1)

print("\nFinal Evaluation Report (BERT-Base, 4-class, 256 tokens, class weights):\n")
print(classification_report(y_true, y_pred, target_names=le.classes_))