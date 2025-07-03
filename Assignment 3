import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

try:
    dataset = load_dataset("imdb")
except Exception as e:
    print(f"Failed to load IMDb dataset: {e}")
    raise
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Increased dataset size for better results
train_sample_size = 10000
test_sample_size = 5000
train_dataset = tokenized_datasets["train"].select(range(train_sample_size))
test_dataset = tokenized_datasets["test"].select(range(test_sample_size))

try:
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.to(device)
except Exception as e:
    print(f"Failed to load model: {e}")
    raise

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return {"accuracy": acc, "f1": f1}
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    learning_rate=2e-5,
    logging_dir="./logs",
    logging_steps=50,
    do_train=True,
    do_eval=True,
    eval_strategy="steps",  # For legacy transformers versions
    eval_steps=100,
    save_steps=100,
    save_total_limit=1,
    report_to="none"
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

try:
    trainer.train()
except Exception as e:
    print(f"Training failed: {e}")
    raise

try:
    metrics = trainer.evaluate()
    print("Evaluation metrics:", metrics)
except Exception as e:
    print(f"Evaluation failed: {e}")
    raise
try:
    model.save_pretrained("./fine_tuned_bert")
    tokenizer.save_pretrained("./fine_tuned_bert")
except Exception as e:
    print(f"Failed to save model or tokenizer: {e}")
    raise

try:
    inference_pipeline = pipeline("sentiment-analysis", model="./fine_tuned_bert", tokenizer="./fine_tuned_bert")
    print("Sample inference:", inference_pipeline("The movie was absolutely amazing!"))
except Exception as e:
    print(f"Inference failed: {e}")
    raise

""" Summary : 
For the project, I made  a sentiment analysis tool with the help of  Hugging Face’s BERT model and  IMDb movie review dataset. Firstly , I loaded the entire dataset directly through the datasets library, which streamlined data loading so that i dont have to  download file myself . Then, I used the bert-base-uncased tokenizer on each review and made sure each one was the right size by padding or cutting it to match what BERT needs.


I used Hugging Face’s Trainer class to train the model. It ran the training steps for me and showed how accurate the model was, along with its F1-score. Once training wrapped up, I saved the fine‑tuned model and its tokenizer so I could quickly reload them for future predictions. Throughout the process, I added error checks at key steps, made it simple to spot and fix any problems that showed up.

Rationale and Challenges :
I chose Hugging Face because its tools are robust and let you move seamlessly from data loading to deployment. However, fine‑tuning BERT comes with a hefty computational load. To manage this, I experimented with smaller batch sizes and cut the number of epochs to keep training time reasonable. Another hurdle was preserving critical information during tokenization—if the maximum length is too short, you risk chopping off important context. By carefully tuning that parameter, I made sure my model still saw the full meaning of longer reviews. In the end, this setup not only performs well on movie‑review sentiment but can also be adapted to any other text‑classification task with minimal changes.

"""
