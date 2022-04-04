#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn


# In[ ]:


import torch
import torch.nn as nn
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AdamW, get_linear_schedule_with_warmup
import math
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import json
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('always')
base_path="/scratch/username/saliency_summaries/"


# In[ ]:


class SaliencyClassifier(nn.Module):

    def __init__(self,
                 nhead=1,
                 nlayers=1,
                 use_cls=True,
                 cls_bail_embed=None,
                 d_model=768):

        super(SaliencyClassifier, self).__init__()

        self.saliency_classifier = nn.Linear(d_model, 2)

        # self.bail_classifier = nn.Linear(d_model, 2)

        ## Use [cls] token or pooling output for bail prediction
        # self.use_cls = use_cls

        # if use_cls:
        #     self.cls_bail_embed = cls_bail_embed ## (1,1,d_model)

        self.encoder_layer = nn.TransformerEncoder(nn.TransformerEncoderLayer(
                                                            d_model=d_model, 
                                                            nhead=nhead,
                                                            batch_first=True), 
                                                      nlayers, 
                                                      norm=None)
  
    def forward(self, x):
        ## x: (batch_size, padded_length, 768)
        batch_size = x.size()[0]
        

        x = self.encoder_layer(x)

        # bail_logits = self.bail_classifier(bail_x)  ## (batch_size, 2) 

        saliency_logits = self.saliency_classifier(x) ## (batch_size, padded_length, 2) 

        return saliency_logits


# In[ ]:


model = SaliencyClassifier(d_model=768)


# In[ ]:


model.load_state_dict(torch.load('sc-all.pt', map_location=torch.device('cuda')))


# In[ ]:


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('device: ' + str(device))


# In[ ]:


model.to(device)


# In[ ]:


def clean_text(row):
    text = []
    [text.extend(i.strip().split('।')) for i in row]
    text = [i.strip() for i in text]
    text = list(filter(None, text))
    return text

def clean_dataset():
    train = pd.read_csv(f'{base_path}train_split_alldistrict_bail.csv')
    test = pd.read_csv(f'{base_path}test_split_alldistricts.csv')
    val = pd.read_csv(f'{base_path}val_split_alldistrict.csv')
    #train=train.head(500)
    #test=test.head(500)
    #val=val.head(500)

    train['ranked-sentences'] = train['ranked-sentences'].apply(eval)
    test['ranked-sentences'] = test['ranked-sentences'].apply(eval)
    val['ranked-sentences'] = val['ranked-sentences'].apply(eval)

    train['segments'] = train['segments'].apply(eval)
    test['segments'] = test['segments'].apply(eval)
    val['segments'] = val['segments'].apply(eval)

    train['ranked-sentences'] = train['ranked-sentences'].apply(clean_text)
    test['ranked-sentences'] = test['ranked-sentences'].apply(clean_text)
    val['ranked-sentences'] = val['ranked-sentences'].apply(clean_text)

    train['facts-and-arguments'] = train['segments'].apply(lambda x: clean_text(x['facts-and-arguments']))
    test['facts-and-arguments'] = test['segments'].apply(lambda x: clean_text(x['facts-and-arguments']))
    val['facts-and-arguments'] = val['segments'].apply(lambda x: clean_text(x['facts-and-arguments']))

    return train, val, test

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        # self.decisions = self.df.decision.map({'dismissed': 0, 'granted': 1})
        self.ranked_sentences = self.df['ranked-sentences']

        self.sentence_model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2')

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        sample = {}
        lines = self.df.iloc[idx]['facts-and-arguments']
        embeddings = self.sentence_model.encode(
            lines
        )

        labels = [0] * len(lines)
        indices = [lines.index(i) for i in self.ranked_sentences.iloc[idx]]
        for i in indices[:len(labels)//2]:
            labels[i] = 1


        sample['embeddings'] = torch.from_numpy(embeddings)
        # sample['bail'] = torch.Tensor([self.decisions.iloc[idx]])
        # sample['salience_labels'] = torch.LongTensor(labels)

        return sample 

def custom_collate(batch):

    # bails, labels, embs = [], [], []
    labels, embs = [], []
    for item in batch:
        # bails.append(item['bail'])
        # labels.append(item['salience_labels'])
        embs.append(item['embeddings'])

    # bails = pad_sequence(bails, batch_first=True)
    embs = pad_sequence(embs, batch_first=True)
    # labels = pad_sequence(labels, padding_value=-100, batch_first=True)
    # return embs, bails.long(), labels.long()
    # return embs, labels.long()
    return embs


# In[ ]:


train, val, test = clean_dataset()


# In[ ]:


train_dataset = Dataset(train)
val_dataset = Dataset(val)
test_dataset = Dataset(test)


# In[ ]:


train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=custom_collate
)

val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=custom_collate
)

test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=custom_collate
)


# In[ ]:


import numpy as np

train_predictions = []
for idx, batch in tqdm(enumerate(train_dataloader),total=len(train_dataloader)):
  with torch.no_grad():
      batch = batch.to(device)
      preds = model(batch).detach().cpu()
      train_predictions.append(preds.tolist())


# In[ ]:


test_predictions = []
for idx, batch in tqdm(enumerate(test_dataloader),total=len(test_dataloader)):
  with torch.no_grad():
      batch = batch.to(device)
      preds = model(batch).detach().cpu()
      test_predictions.append(preds.tolist())


# In[ ]:


val_predictions = []
for idx, batch in tqdm(enumerate(val_dataloader),total=len(val_dataloader)):
  with torch.no_grad():
      batch = batch.to(device)
      preds = model(batch).detach().cpu()
      val_predictions.append(preds.tolist())


# In[ ]:


import pickle
with open('/scratch/username/o1/train_preds.pkl', 'wb') as f:
  pickle.dump(train_predictions, f)

with open('/scratch/username/o1/test_preds.pkl', 'wb') as f:
  pickle.dump(test_predictions, f)

with open('/scratch/username/o1/val_preds.pkl', 'wb') as f:
  pickle.dump(val_predictions, f)


# In[ ]:




