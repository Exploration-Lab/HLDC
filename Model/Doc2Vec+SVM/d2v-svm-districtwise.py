#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import re
import codec
import argparse
import logging
import shutil
import json
from random import shuffle, randint
from datetime import datetime
from collections import namedtuple, OrderedDict
import multiprocessing
from smart_open import open
from tqdm.auto import tqdm
import gensim
import gensim.models.doc2vec
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import time
import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import pickle


# In[ ]:


def doc2vec(X_train,y_train,X_test,y_test,svc_c=0,trial=None,num_epochs=0,alpha=0.015):
  stopwords_path='/home2/usrname/legal-tech/hindi_stop.txt'
  vocab_min_count=5
  if trial!=None:
      num_epochs=trial.suggest_int("num_epochs", 50, 100)
      alpha = trial.suggest_float("alpha", 0.005, 0.05)
  algorithm="pv_dmc"
  vector_size=200
  min_alpha=0.001
  window=5
  negative = 5
  hs = 0
  def read_lines(path):
    return [line.strip() for line in codecs.open(path, "r", "utf-8")]
  def load_stopwords(stopwords_path):
    stopwords = read_lines(stopwords_path)
    return dict(map(lambda w: (w.lower(), ''), stopwords))
  assert gensim.models.doc2vec.FAST_VERSION > -         1, "This will be painfully slow otherwise"
  stopwords = load_stopwords(stopwords_path)
  cores = multiprocessing.cpu_count()
  docs=[]
  for i , doc in enumerate(X_train):
    words = doc.replace("\n"," ").replace("।", " ")
    words = re.sub(r'[^\w\s]', " ", words).split()
    words = [w for w in words if w not in stopwords and len(w) > 1]
    tags=[i]
    docs.append(TaggedDocument(words=words, tags=tags))
  if algorithm == 'pv_dmc':
        model = Doc2Vec(dm=1, dm_concat=1, vector_size=vector_size, window=window, negative=negative, hs=hs,
                        min_count=vocab_min_count, workers=cores)
  elif algorithm == 'pv_dma':
      model = Doc2Vec(dm=1, dm_mean=1, vector_size=vector_size, window=window, negative=negative, hs=hs,
                      min_count=vocab_min_count, workers=cores)
  elif algorithm == 'pv_dbow':
      model = Doc2Vec(dm=0, vector_size=vector_size, window=window, negative=negative, hs=hs,
                      min_count=vocab_min_count, workers=cores)
  vocab_size = len(model.wv.index_to_key)
  model.build_vocab(docs)
  shuffle(docs)
  print("Training")
  model.train(docs, total_examples=len(docs),
              epochs=num_epochs, start_alpha=alpha, end_alpha=min_alpha,report_delay=60)
  Xtr=[]
  for i , doc in enumerate(X_train):
    Xtr.append(model.dv.get_vector(i))
  Xte=[]
  for i , doc in enumerate(X_test):
    words = doc.replace("\n"," ").replace("।", " ")
    words = re.sub(r'[^\w\s]', " ", words).split()
    words = [w for w in words if w not in stopwords and len(w) > 1]
    Xte.append(model.infer_vector(words))
  from sklearn.svm import SVC
  print("Classifying")
  if trial!=None:
      svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
  clf = SVC(C=svc_c, gamma="auto")
  clf.fit(Xtr, y_train)
  from sklearn.metrics import classification_report
  y_pred = clf.predict(Xte)
  if trial==None:
    plot_confusion_matrix(clf, Xte, y_test)
    plt.savefig("/home2/usrname/legal-tech/doc2vec-dw-svm.png",dpi=300)
    with open("./doc2vec-dw-svm.pkl","wb") as f:
        pickle.dump(clf,f)
    model.save("./doc2vec-dw-svm.model")
  return classification_report(y_test, y_pred,output_dict=True)


# In[ ]:


with open(f"/scratch/usrname/train_test_data_for_modelling/test_split_17_districts.json") as f:
    data = json.load(f)
    df = pd.DataFrame(data)
    #df = df.head(100)
    df2 = df.sample(frac = 0.1, random_state=42).reset_index()


# In[ ]:


X_test = df["segments"].apply(lambda x:" ".join(x["facts-and-arguments"])).tolist()
y_test = df["decision"].apply(lambda x: 1 if x == "granted" else 0).tolist()
hX_test = df2["segments"].apply(lambda x:" ".join(x["facts-and-arguments"])).tolist()
hy_test = df2["decision"].apply(lambda x: 1 if x == "granted" else 0).tolist()


# In[ ]:


with open(f"/scratch/usrname/train_test_data_for_modelling/train_split_44_districts.json") as f:
    data = json.load(f)
    df = pd.DataFrame(data)
    #df = df.head(100)
    df2 = df.sample(frac = 0.1, random_state=42).reset_index()


# In[ ]:


X_train = df["segments"].apply(lambda x:" ".join(x["facts-and-arguments"])).tolist()
y_train = df["decision"].apply(lambda x: 1 if x == "granted" else 0).tolist()
hX_train = df2["segments"].apply(lambda x:" ".join(x["facts-and-arguments"])).tolist()
hy_train = df2["decision"].apply(lambda x: 1 if x == "granted" else 0).tolist()


# In[ ]:


def objective(trial):
    rep = doc2vec(hX_train,hy_train,hX_test,hy_test,svc_c=None,trial=trial,num_epochs=0,alpha=0.015)
    accuracy = rep["accuracy"]
    return accuracy


# In[ ]:


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)
print(study.best_trial)


# In[ ]:


trial = study.best_trial
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
    if key=="svc_c":
        svc_c=value
    if key=="num_epochs":
        num_epochs=value
    if key=="alpha":
        alpha=value


# In[ ]:


rep = doc2vec(X_train,y_train,X_test,y_test,svc_c=svc_c,trial=None,num_epochs=num_epochs,alpha=alpha)
print(rep)


# In[ ]:


with open("/home2/usrname/legal-tech/doc2vec-dw-svm.json","w") as f:
    json.dump(rep,f,indent=4)


# In[ ]:




