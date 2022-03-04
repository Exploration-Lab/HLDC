from tqdm import tqdm
import re
import string

import numpy as np
import networkx as nx
from nltk import word_tokenize, sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json

from indicnlp.tokenize import indic_tokenize

import os
files = os.listdir("path to folder/legal_train_test_data/combined_data/")

import pickle
file = open("path to folder/textRank/model_all_districts/words_final.pkl",'rb')
corpus = pickle.load(file)
file.close()
print("Corpus Loaded")

from gensim.models import FastText
model = FastText.load('path to folder/textRank/model_all_districts/model_final.bin')
print("Model Loaded")

def process(file):
    print(f"Current file:{file}")
    with open(f"path to folder/legal_train_test_data/combined_data/{file}") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    ranked_sentences = []
    for i, row in tqdm(df.iterrows(),total=len(df)):
        src_sents = df.loc[i]["segments"]['facts-and-arguments']
        src_sents = [i.split('।') for i in src_sents]
        # split all paragraphs in individual sentences
        src_sents = [i for i in src_sents if len(i)!=0 and i!=' ']
        src_sents = [i for subl in src_sents for i in subl]
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
        sentence_vectors = []
        sentences = [elt for elt in sentences if elt != ' ']
        n = len(sentences)
        for sent in sentences:
          if len(i) != 0:
            tokens = indic_tokenize.trivial_tokenize(sent)
            try:
              v = sum([model.wv[w] for w in tokens])/(len(tokens) + 0.001)
            except KeyError:
              print('sent', sent)
              print('tokens', tokens)
          else:
              v = np.zeros((100,))
          sentence_vectors.append(v)

        # similarity matrix
        sim_mat = np.zeros([n, n])
        for i in range(n):
          for j in range(n):
              if i != j:
                  sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank_numpy(nx_graph)
        ranked_sentences.append(sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True))

    print(len(df), len(ranked_sentences))
    df = df.head(len(ranked_sentences))
    df['ranked-sentences'] = [i[:10] for i in ranked_sentences]
    df['ranked-sentences'].map(len)
    
    file = file.replace(".json",".csv")
    df.to_csv(f"path to folder/textRank_summaries/{file}")
    
process("test_split_alldistricts.json")
