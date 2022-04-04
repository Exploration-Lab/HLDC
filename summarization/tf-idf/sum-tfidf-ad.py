#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tqdm.auto import tqdm
import re
import string


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


corpus = []
vectoriser = TfidfVectorizer()


# In[ ]:


import json
import pandas as pd
def put_in_corpus(file):
  global corpus
  print(f"Current file:{file}")
  with open(f"/scratch/username/train_test_data_for_modelling/{file}") as f:
      data = json.load(f)
  df = pd.DataFrame(data)
  for i, row in tqdm(df.iterrows(),total=len(df)):
      src_sents=[]
      paras = df.loc[i]["segments"]['facts-and-arguments']
      for para in paras:
        sent = para.split('।')
        sent = [i for i in sent if len(i)!=0 and i!=' ']
        src_sents.extend(sent)
      src_sents = list(filter(None, src_sents))
      src_sents1=[]
      for sent in src_sents:
        try:
          sent = ''.join([i for i in sent if not i.isdigit()])
        except:
          print(sent)
        src_sents1.append(sent)
      src_sents =src_sents1
      corpus.extend(src_sents)


# In[ ]:


import os
files = os.listdir("/scratch/username/train_test_data_for_modelling/")


# In[ ]:


try:
    os.mkdir("/scratch/username/summaries_octtfidf2")
except:
    pass


# In[ ]:


files = [f for f in files if ".json" in f]


# In[ ]:


files = ["train_split_alldistrict_bail.json"]
for file in files:
    put_in_corpus(file)


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
def tok(s):
  return s.split()
cv = CountVectorizer(tokenizer=tok)
data2 = cv.fit_transform(corpus)
tfidf_transformer = TfidfTransformer()
tfidf_matrix = tfidf_transformer.fit_transform(data2)
word2tfidf = dict(zip(cv.get_feature_names(), tfidf_transformer.idf_))


# In[ ]:


def get_score(sentence):
  words = sentence.split()
  score = 0
  for word in words:
    try:
      score+= word2tfidf[word]
    except:
      pass
  return score


# In[ ]:


def process(file):
    print(f"Current file:{file}")
    with open(f"/scratch/username/train_test_data_for_modelling/{file}") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    ranked_sentences = []
    for i, row in tqdm(df.iterrows(),total=len(df)):
        src_sents = df.loc[i]["segments"]['facts-and-arguments']
        src_sents = [i.split('।') for i in src_sents]
        # split all paragraphs in individual sentences
        src_sents = [i for subl in src_sents for i in subl]
        src_sents = [i for i in src_sents if len(i)!=0 and i!=' ']
        src_sents = list(filter(None, src_sents))
        src_sents1=[]
        for sent in src_sents:
          try:
            sent = ''.join([i for i in sent if not i.isdigit()])
          except:
            print(sent)
          src_sents1.append(sent)
        src_sents =src_sents1
        sentences = src_sents
        scores=[]
        for sent in sentences:
          scores.append(get_score(sent))
        ranks = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
        ranks = [i[1] for i in ranks]
        ranked_sentences.append(ranks)
    df = df.head(len(ranked_sentences))
    df['ranked-sentences'] = [i[:10] for i in ranked_sentences]
    df['ranked-sentences'].map(len)
    file = file.replace(".json",".csv")
    df.to_csv(f"/scratch/username/summaries_octtfidf2/{file}")


# In[ ]:


files = ["train_split_alldistrict_bail.json","test_split_alldistricts.json","val_split_alldistrict.json"]
for file in files:
  process(file)


# In[ ]:




