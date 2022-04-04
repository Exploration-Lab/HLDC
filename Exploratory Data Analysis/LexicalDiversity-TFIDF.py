#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import glob
import pickle
import ast
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from multiprocessing.pool import ThreadPool as Pool


# In[ ]:


districts = ['agra', 'aligarh', 'varanasi', 'lucknow', 'ghaziabad', 'ambedkar_nagar', 'bahraich', 'azamgarh',
             'allahabad', 'balrampur', 'auraiya', 'barabanki', 'banda', 'bagpat', 'bhadohi', 'ballia', 'bijnor',
             'basti', 'bareilly', 'bulandshahar', 'chitrakoot', 'deoria', 'budaun', 'etah', 'etawah', 'farrukhabad',
             'faizabad', 'fatehpur', 'firozabad', 'gautam_buddha_nagar', 'ghazipur', 'hapur', 'gonda', 'hardoi',
             'hamirpur_up', 'jaunpur', 'gorakhpur', 'jalaun', 'jyotiba_phule_nagar', 'jhansi', 'hathras', 'kanpur_dehat',
             'kanpur_nagar', 'kannauj', 'kanshiramnagar', 'kheri', 'kaushambi', 'kushinagar', 'lalitpur', 'maharajganj',
             'mainpuri', 'meerut', 'mahoba', 'mirzapur', 'mathura', 'moradabad', 'muzaffarnagar', 'pratapgarhdistrict', 'pilibhit',
             'rampur', 'raebareli', 'mau', 'saharanpur', 'sant_kabir_nagar', 'shahjahanpur', 'shravasti', 'siddharthnagar', 'sitapur', 'unnao',
             'sultanpur', 'sonbhadra']


# In[ ]:


assert len(districts) == 71
HOME = "../all_bail_cases_pickles/"
def file(district):
  return f"{HOME}/{district}/full_data_after_simple_NER_division.json"


# In[ ]:


text_per_district = {}

for district in tqdm(districts):
  with open(file(district), 'r') as f:
    data = json.load(f)
#   print(district)
  text = ""
  for court in data.keys():
    df = pd.DataFrame(data[court]['processed']).T
    for idx, i in df.iterrows():
      text += i['header'] + " ".join(i['body']) + i['result']
    for i in data[court]['valid']:
        text += data[court]['valid'][i]
  text_per_district[district] = text


# In[ ]:


len(text_per_district)


# In[ ]:


pickle.dump(text_per_district, open("textperdistrict", "wb"))


# In[ ]:


text_per_district = pickle.load(open("textperdistrict", "rb"))


# In[ ]:


len(text_per_district)


# In[ ]:


sent_count = 0
for dst in text_per_district:
    sent_count += text_per_district[dst].count("।")


# In[ ]:


import string


# In[ ]:


updated_text = {}

for i in tqdm(text_per_district):
    txt = text_per_district[i]
    txt = txt.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    txt = txt.translate(str.maketrans(string.digits, ' '*len(string.digits)))
    updated_text[i] = txt


# In[ ]:


corpus = list(updated_text.values())


# In[ ]:


import re


# In[ ]:


corpus[0][:1000]


# In[ ]:


re.findall(r'[^(\s|।||)]+', corpus[0][:1000])


# In[ ]:


get_ipython().system('pip install sklearn')


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


vectorizer = TfidfVectorizer(token_pattern=r'[^(\s|।||)]+', stop_words=stopwords)


# In[ ]:


X = vectorizer.fit_transform(corpus)


# In[ ]:


X.shape


# In[ ]:


for i in districts:
    txt = updated_text[i]
    response = vectorizer.transform([txt])
    feature_array = np.array(vectorizer.get_feature_names())
    tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]
    n = 50
    top_n = feature_array[tfidf_sorting][:n]
    print(i, top_n)

