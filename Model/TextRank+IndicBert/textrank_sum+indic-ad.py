#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from torch.utils.data import Dataset
import torch
import os
import json
import re
from tqdm import tqdm
tqdm.pandas()
from transformers import Trainer, TrainingArguments
import numpy as np
from datasets import load_metric
from sklearn.model_selection import train_test_split
import ast


# In[ ]:


tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")


# In[ ]:


def model_init():
    return AutoModelForSequenceClassification.from_pretrained("ai4bharat/indic-bert", num_labels=2)


# In[ ]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# In[ ]:


train_df = pd.read_csv("/scratch/username/textrank_summaries/train_split_alldistrict_bail.csv")
test_df = pd.read_csv("/scratch/username/textrank_summaries/val_split_alldistrict.csv")
#train_df = train_df.head(500)
#test_df = test_df.head(500)
hp_train_df = train_df.sample(frac = 0.1, random_state=42).reset_index()
hp_test_df = test_df.sample(frac = 0.1, random_state=42).reset_index()


# In[ ]:


class LegalDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df.reset_index(drop=True)
        self.df["text"] = self.df["ranked-sentences"].progress_apply(lambda x:" ".join([i[1] for i in eval(x)[:10]]))
        self.df["label"] = self.df["decision"].progress_apply(lambda x:1 if x=="granted" else 0)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        model_input = self.df['text'][idx]            
        encoded_sent = self.tokenizer.encode_plus(
            text=model_input, 
            add_special_tokens=True,       
            max_length=512,                  
            padding='max_length',          
            return_attention_mask=True, 
            truncation=True
            )
        
        input_ids = encoded_sent.get('input_ids')
        attention_mask = encoded_sent.get('attention_mask')
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)        

        label = torch.tensor(self.df['label'][idx])
        
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'label': label}


# In[ ]:


train_dataset = LegalDataset(train_df, tokenizer)
test_dataset = LegalDataset(test_df, tokenizer)
hp_train_dataset = LegalDataset(hp_train_df, tokenizer)
hp_test_dataset = LegalDataset(hp_test_df, tokenizer)


# In[ ]:


metric1 = load_metric("accuracy")
metric2 = load_metric("f1")


# In[ ]:


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = metric1.compute(predictions=predictions, references=labels)
    f1 = metric2.compute(predictions=predictions, references=labels, average="micro")
    return {'accuracy': accuracy["accuracy"], 'f1-score': f1["f1"]}


# In[ ]:


def my_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "weight_decay":trial.suggest_float("weight_decay", 0.005, 0.05),
        "adam_beta1":trial.suggest_float("adam_beta1", 0.75, 0.95),
        "adam_beta2":trial.suggest_float("adam_beta2", 0.99, 0.9999),
        "adam_epsilon":trial.suggest_float("adam_epsilon", 1e-9, 1e-7, log=True)
    }


# In[ ]:


training_args = TrainingArguments(
    output_dir='/scratch/username/htr2_results',          # output directory
    num_train_epochs=5,            # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=8,   # batch size for evaluation
    warmup_steps=500,               # number of warmup steps for learning rate scheduler
    weight_decay=0.01,              # strength of weight decay
    logging_dir='/scratch/username/htr2_logs',           # directory for storing logs
    evaluation_strategy="epoch",
    logging_steps=250,
    save_strategy='epoch',
    save_total_limit = 1,
    learning_rate = 0.00001,
    load_best_model_at_end=True,
    metric_for_best_model ="eval_f1-score",
)


# In[ ]:


trainer = Trainer(
    model_init=model_init,                        # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=hp_train_dataset,         # training dataset
    eval_dataset=hp_test_dataset,           # evaluation dataset
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)


# In[ ]:


best_run = trainer.hyperparameter_search(n_trials=10,direction="maximize",hp_space=my_hp_space)


# In[ ]:


print("Best HyperParameters")


# In[ ]:


print(best_run)


# In[ ]:


del trainer
del training_args
import gc
gc.collect()


# In[ ]:


print("Starting Training...")


# In[ ]:


training_args = TrainingArguments(
    output_dir='/scratch/username/tr2_results',          # output directory
    num_train_epochs=15,            # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=8,   # batch size for evaluation
    warmup_steps=500,               # number of warmup steps for learning rate scheduler
    weight_decay=0.01,              # strength of weight decay
    logging_dir='/scratch/username/tr2_logs',           # directory for storing logs
    evaluation_strategy="epoch",
    logging_steps=250,
    save_strategy='epoch',
    save_total_limit = 1,
    learning_rate = 0.00001,
    load_best_model_at_end=True,
    metric_for_best_model ="eval_f1-score",
)


# In[ ]:


trainer = Trainer(
    model_init=model_init,                        # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,           # evaluation dataset
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)


# In[ ]:


for n, v in best_run.hyperparameters.items():
    setattr(trainer.args, n, v)
print(trainer.args)
trainer.train()


# In[ ]:


trainer.save_model("/home2/username/legal-tech/textrank_sum+indic-ad")


# In[ ]:




