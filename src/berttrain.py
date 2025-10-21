import asyncio
from transformers import DistilBertTokenizerFast, DistilBertModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datasets import Dataset
from transformers import AutoTokenizer


CACHE_DIR = "./models"
DEVICE="cuda"

df = pd.read_csv("training_data/GateClassifier/friday_agent_500_binary.csv")
dataset = Dataset.from_pandas(df)



MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

tokenized = dataset.map(tokenize, batched=True)

from datasets import DatasetDict

tokenized = tokenized.rename_column("label", "labels")
splits = tokenized.train_test_split(test_size=0.2, seed=42)
train_ds = splits["train"]
eval_ds  = splits["test"]

from transformers import AutoModelForSequenceClassification

num_labels = len(set(df["label"]))
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
)
model.to("cuda")

from transformers import TrainingArguments, Trainer
import evaluate

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return accuracy.compute(predictions=preds, references=labels)

training_args = TrainingArguments(
    output_dir="./results",
    do_eval=True,                # run eval during training
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    eval_steps=50,               # evaluate every 50 steps
    save_steps=50,               # save every 50 steps
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model("models/friday_agent_model")
tokenizer.save_pretrained("models/friday_agent_model")


metrics = trainer.evaluate()
print(metrics)
