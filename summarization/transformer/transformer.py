#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import tqdm
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')


# In[ ]:


def process(file):
    print(f"Current file:{file}")
    with open(f"/scratch/username/train_test_data_for_modelling/{file}") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    #df = df.head(50)
    ranked_sentences = []
    for i, row in tqdm.notebook.tqdm(df.iterrows(),total=len(df)):
        src_sents=[]
        paras = df.loc[i]["segments"]['facts-and-arguments']
        for para in paras:
            sent = para.split('।')
            sent = [i for i in sent if len(i)!=0 and i!=' ']
            src_sents.extend(sent)
        s1=src_sents
        s2 = [" ".join(df.loc[i]["segments"]['judge-opinion'])]
        e1 = model.encode(s1)
        e2 = model.encode(s2)
        cos_sim = util.cos_sim(e2, e1)
        s1 = np.asarray(s1)
        ranks=s1[np.argsort(-1*cos_sim[0].numpy())]
        ranked_sentences.append(ranks.tolist())
    df = df.head(len(ranked_sentences))
    df['ranked-sentences'] = ranked_sentences
    df['ranked-sentences'].map(len)
    file = file.replace(".json",".csv")
    df.to_csv(f"/scratch/username/saliency_summaries/{file}")


# In[ ]:


files = os.listdir("/scratch/username/train_test_data_for_modelling/")


# In[ ]:


try:
    os.mkdir("/scratch/username/saliency_summaries")
except:
    pass


# In[ ]:


files = [f for f in files if ".json" in f]


# In[ ]:


for file in files:
    process(file)


# In[ ]:




