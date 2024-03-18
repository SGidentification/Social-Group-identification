# !pip install datasets evaluate transformers[sentencepiece]
# !pip install accelerate
# !apt install git-lfs
# !pip install seqeval
import random
random.seed(42)
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, get_scheduler
from transformers import TrainingArguments, Trainer
from seqeval.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import evaluate
from torch.optim import AdamW
from accelerate import Accelerator
#model_checkpoint ='bert-large-cased'
model_checkpoint ='roberta-large'
import os
import numpy as np
import pandas as pd
labels_to_ids = {"O": 0, "I":1, "B":2}
ids_to_labels = {0: "O", 1: "I", 2:"B"}
label2id = {"O": 0, "I-S":1, "B-S":2}
id2label = {0: "O", 1: "I-S", 2:"B-S"}

train_files = [
    "./TOKENIZED_BIO_random_samples_test_Economics_annotated_gold.csv",
    "./TOKENIZED_BIO_random_samples_test_worldpolitics_annotated_gold.csv",
    "./TOKENIZED_BIO_random_samples_test_politics_annotated_gold.csv"
]
test_file = "./TOKENIZED_BIO_random_samples_test_ukpolitics_annotated_gold.csv"
validation_file = "./TOKENIZED_BIO_random_samples_test_Libertarian_annotated_gold.csv"

#tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
#for RoBERTa uncomment below line
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=3, id2label=id2label, label2id=label2id, finetuning_task='ner')
data_collator = DataCollatorForTokenClassification(tokenizer)

label_list = ['O', 'B-S', 'I-S']

# Apply postprocessing to each element in the array
def convert_to_bio_format(seq):

    bio_sequence = (len(seq))*["O"]
    inside_entity = False
    for i, token in enumerate(seq):
        if token.startswith("@@") and token.endswith("##"):
            bio_sequence[i] = "B"
            inside_entity = False
        elif token.startswith("@@"):
            bio_sequence[i] = "B"
            inside_entity = False
            next_at_index = -1
            next_hash_index = -1

            # Find the index of the next token starting with @@
            for j, t in enumerate(seq[i + 1:]):
                if t.startswith("@@"):
                    next_at_index = i + 1 + j
                    break

            # Find the index of the next token ending with ##
            for j, t in enumerate(seq[i + 1:]):
                if t.endswith("##"):
                    next_hash_index = i + 1 + j
                    break

            if next_at_index != -1 and next_hash_index != -1 and next_hash_index < next_at_index:
                inside_entity = True
            if next_at_index == -1 and next_hash_index != -1:
                inside_entity = True
        elif inside_entity:
            bio_sequence[i] = "I"
            if not token.endswith("##"):
                inside_entity = True
            else:
                inside_entity = False
    return bio_sequence

def convert_bio_to_ids(bio_sequence):
    return [labels_to_ids[label] for label in bio_sequence]

# Function to apply 'convert_to_bio_format' to a sequence
def apply_conversion(seq):
    bio_sequence = convert_to_bio_format(seq.split()) 
    ids_sequence = convert_bio_to_ids(bio_sequence)
    return bio_sequence, ids_sequence



# Function to process a single CSV file
def process_csv_file(file_path):
    df = pd.read_csv(file_path, sep='\t')
    df['tokenized_body_cleaned'] = df['tokenized_body_cleaned'].apply(eval) 
    df['bio_annotation'] = df['bio_annotation'].apply(eval) 
    df['ids'] = df['ids'].apply(eval) 
    dataset = Dataset.from_pandas(df)
    return dataset

# Folder containing the CSV files
folder_path = "/."
# for filename in os.listdir(folder_path):
#     if filename.endswith(".csv"):
#         input_file_path = os.path.join(folder_path, filename)
#         output_file_path = os.path.join(folder_path, "BIO_" + filename)  # Output file path
#         process_csv_file(input_file_path, output_file_path)

def process_all_train_files(train_files):
    train_datasets = []
    for file_path in train_files:
        df = pd.read_csv(file_path, sep='\t')
        df['tokenized_body_cleaned'] = df['tokenized_body_cleaned'].apply(eval)  
        df['bio_annotation'] = df['bio_annotation'].apply(eval)  
        df['ids'] = df['ids'].apply(eval)  
        train_datasets.append(Dataset.from_pandas(df))
    return train_datasets

# Create train, test, validation sets
train_datasets = process_all_train_files(train_files)
train_dataset = concatenate_datasets(train_datasets)
test_dataset = process_csv_file(test_file)
validation_dataset = process_csv_file(validation_file)


# Create DatasetDict
data_dict = DatasetDict({
    "train": train_dataset,
    "test": test_dataset,
    "validation": validation_dataset
})

def tokenize_and_align_labels(examples, label_all_tokens=True):
    tokenized_inputs = tokenizer(examples["tokenized_body_cleaned"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ids"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx >= len(label):
                label_ids.append(-100)
            else:
                if word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

## Applying on entire data
tokenized_datasets = data_dict.map(tokenize_and_align_labels, batched=True)

#F1, precision, recall
def compute_metrics(eval_preds):
    pred_logits, labels = eval_preds
    pred_logits = np.argmax(pred_logits, axis=2)
    predictions = [
        [label_list[eval_preds] for (eval_preds, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]

    true_labels = [
      [label_list[l] for (eval_preds, l) in zip(prediction, label) if l != -100]
       for prediction, label in zip(pred_logits, labels)
   ]

    f1 = f1_score(y_true=true_labels, y_pred=predictions,  mode="strict", average='micro')
    precision = precision_score(y_true=true_labels, y_pred=predictions,  mode="strict", average='micro')
    recall = recall_score(y_true=true_labels, y_pred=predictions,  mode="strict", average='micro')

    return {
          "precision": precision,
          "recall": recall,
          "f1": f1,
  }

args = TrainingArguments("SocialGroups", evaluation_strategy = "epoch", learning_rate=5e-5, per_device_train_batch_size=4, per_device_eval_batch_size=4, num_train_epochs=4, weight_decay=0.1,)
trainer = Trainer( model, args, train_dataset=tokenized_datasets["train"], eval_dataset=tokenized_datasets["validation"], data_collator=data_collator, tokenizer=tokenizer, compute_metrics=compute_metrics)
trainer.train()
trainer.evaluate(eval_dataset=tokenized_datasets["test"])
