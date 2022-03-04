import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from sklearn import preprocessing
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoConfig, AdamW
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizer
from torch.utils.data import Dataset
import torch
from torch import nn

from sklearn.model_selection import train_test_split 
import json



def getDistrictData(fileName):
    
    with open(fileName) as f:
        data = json.load(f)
    
    data = data['processed']
    
    for key in data.keys():

        if data[key]['decision'] == 'dismissed':
            data[key]['label'] = 0
        if data[key]['decision'] == 'granted':
            data[key]['label'] = 1
    
    return data 

with open("/scratch/username/train_test_data_for_modelling/train_split_alldistrict_bail.json") as f:
    data = json.load(f)
df_train = pd.DataFrame(data)


with open("/scratch/username/train_test_data_for_modelling/val_split_alldistrict.json") as f:
    data = json.load(f)
df_validation = pd.DataFrame(data)


df_train['decision'].describe()
df_train['decision'] = (df_train['decision'] == 'granted').astype(int)
df_validation['decision'] = (df_validation['decision'] == 'granted').astype(int)



df_train['decision'].describe()
df_validation['decision'].describe()



tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
model = AutoModelForSequenceClassification.from_pretrained("ai4bharat/indic-bert", num_labels=2)


class CaseDataset(Dataset):
    def __init__(self,data,tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)-1
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Create empty lists to store outputs
        input_ids = []
        attention_mask = []

        encoded_sent = self.tokenizer.encode_plus(
            text=self.data.iloc[idx]['segments']['facts-and-arguments'][0],   # Preprocess sentence
            add_special_tokens=True,
            max_length=512,                  # Max length to truncate/pad
            padding='max_length',            # Pad sentence to max length
            return_attention_mask=True,      # Return attention mask
            truncation=True
            )
        
        input_ids = encoded_sent.get('input_ids')
        attention_mask = encoded_sent.get('attention_mask')
        
        ## Take the first 512 tokens 
        input_ids = input_ids[:512]
        attention_mask = attention_mask[:512]
        
        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)        

        label = torch.tensor(self.data.iloc[idx]['decision'])
        
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'label': label}



train_dataset = CaseDataset(df_train, tokenizer)
validation_dataset = CaseDataset(df_validation, tokenizer)


from transformers import Trainer, TrainingArguments
import numpy as np
from datasets import load_metric


metric1 = load_metric("accuracy")
metric2 = load_metric("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = metric1.compute(predictions=predictions, references=labels)
    f1 = metric2.compute(predictions=predictions, references=labels)
    return {'acuracy': accuracy, 'f1-score': f1}



training_args = TrainingArguments(
    output_dir='/scratch/username/results_first_512_parameter_all_district',          # output directory
    num_train_epochs=5,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=8,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='/scratch/username/logs_first_512_parameter_all_district',            # directory for storing logs
    evaluation_strategy="epoch",
    logging_steps=250,
    save_strategy='epoch',
    learning_rate = 1e-5,
    save_total_limit=1
)



def model_init():
    return AutoModelForSequenceClassification.from_pretrained("ai4bharat/indic-bert", num_labels=2)


hyper_paramter_train_dataset = CaseDataset(df_train.sample(frac = 0.1, random_state=42), tokenizer)
hyper_paramter_val_dataset = CaseDataset(df_validation.sample(frac = 0.1, random_state=42), tokenizer)



def compute_metrics_hyperparameter(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = metric1.compute(predictions=predictions, references=labels)
    f1 = metric2.compute(predictions=predictions, references=labels)
    return f1

def my_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "weight_decay":trial.suggest_float("weight_decay", 0.005, 0.05), 
        "adam_beta1":trial.suggest_float("adam_beta1", 0.75, 0.95),
        "adam_beta2":trial.suggest_float("adam_beta2", 0.99, 0.9999),
        "adam_epsilon":trial.suggest_float("adam_epsilon", 1e-9, 1e-7, log=True) 
    }


trainer = Trainer(
    model_init=model_init,                    # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=hyper_paramter_train_dataset,         # training dataset
    eval_dataset=hyper_paramter_val_dataset,     # evaluation dataset
    compute_metrics=compute_metrics_hyperparameter
)


best_run = trainer.hyperparameter_search(n_trials=10, direction="maximize", hp_space=my_hp_space)



del trainer
del training_args
import gc
gc.collect()


training_args = TrainingArguments(
    output_dir='/scratch/username/results_first_512_parameter_all_district',          # output directory
    num_train_epochs=10,              # total number of training epochs
    per_device_train_batch_size=8,    # batch size per device during training
    per_device_eval_batch_size=8,     # batch size for evaluation
    warmup_steps=500,                 # number of warmup steps for learning rate scheduler
    weight_decay=0.01,                # strength of weight decay
    logging_dir='/scratch/username/logs_first_512_parameter_all_district',            # directory for storing logs
    evaluation_strategy="epoch",
    logging_steps=250,
    save_strategy='epoch',
    learning_rate = 1e-5,
    save_total_limit=1
)


final_trainer = Trainer(
    model_init=model_init,                    # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=validation_dataset,     # evaluation dataset
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)


for n, v in best_run.hyperparameters.items():
    setattr(final_trainer.args, n, v)


final_trainer.train()
final_trainer.save_model("/home2/username/courts/final_models/first_512_all_district")
