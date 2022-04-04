#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from os import path
from tqdm import tqdm 


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
# districts = ['ghaziabad', 'agra', 'varanasi', 'aligarh', 'kanpurnagar', 'lucknow']
assert len(districts) == 71
def file(district):
  return f"../all_bail_cases_pickles/{district}/full_data_after_simple_NER_division.json"


# In[ ]:


HOME_TT = "pathTo/legal_train_test_data/combined"


# In[ ]:


get_ipython().system('ls $HOME_TT')


# In[ ]:


import pandas as pd


# In[ ]:


def file(district):
  return f"{HOME_TT}/{district}/full_data_after_simple_NER_division.json"


# In[ ]:


from tqdm import tqdm


# # SPLIT

# In[ ]:


court_complexes = df.index
for i in court_complexes:
  df_complex = pd.DataFrame(df.loc[i]['processed']).T
  if len(df_complex) == 0:
    continue
  df_complex = df_complex[df_complex['decision'] != "don't know"]
  df_complex.dropna(subset=['segments'])


# In[ ]:


c1.dropna(subset=['segments'])


# In[ ]:


c1 = pd.DataFrame(df.loc[court_complexes[0]]['processed']).T
c1 = c1[c1['decision'] != "don't know"]
c1


# In[ ]:


district_wise_useful_counts = []


# In[ ]:


for district in tqdm(districts):
  filename = file(district)
  count = 0
  df = pd.read_json(filename, orient = 'index')
  court_complexes = df.index
  for i in court_complexes:
    df_complex = pd.DataFrame(df.loc[i]['processed']).T
    if len(df_complex) == 0 or 'segments' not in df_complex:
      continue
    df_complex = df_complex[df_complex['decision'] != "don't know"]
    if len(df_complex) == 0:
      continue
    df_complex = df_complex.dropna(subset=['segments'])
    count += len(df_complex)
  district_wise_useful_counts.append((district, count))


# In[ ]:


district_wise_useful_counts


# In[ ]:


import random


# In[ ]:


random.seed(42)
random.shuffle(district_wise_useful_counts)


# In[ ]:


train_count = 123742
test_count = 35400
validation_count = 17707


# In[ ]:


train_districts = []
test_districts = []
validation_districts = []

current_count = train_count
current_ptr = train_districts
for i in district_wise_useful_counts:
    cnt = i[1]
    current_count -= cnt
    current_ptr.append(i)
    if current_count < 500:
        if current_ptr is train_districts:
            current_count = test_count
            current_ptr = test_districts
        elif current_ptr is test_districts:
            current_count = validation_count
            current_ptr = validation_districts


# In[ ]:


len(train_districts), len(test_districts), len(validation_districts)


# In[ ]:





# In[ ]:


validation_districts


# In[ ]:


global_training_complete_df = []
for district in tqdm(validation_districts):
  district = district[0]
  filename = file(district)
  df = pd.read_json(filename, orient = 'index')
  court_complexes = df.index
  district_df = pd.DataFrame()
  for i in court_complexes:
    df_complex = pd.DataFrame(df.loc[i]['processed']).T
    if len(df_complex) == 0 or 'segments' not in df_complex:
      continue
    df_complex = df_complex[df_complex['decision'] != "don't know"]
    if len(df_complex) == 0:
      continue
    df_complex = df_complex.dropna(subset=['segments'])
    if len(df_complex)!=0:
        df_complex['complex'] = i
        district_df = pd.concat((district_df, df_complex))
  district_df['district'] = district
  global_training_complete_df.append(district_df)


# In[ ]:


global_training_complete_df = pd.concat(global_training_complete_df)


# In[ ]:


global_training_complete_df['case_number'] = global_training_complete_df.index


# In[ ]:


global_training_complete_df.reset_index(inplace=True)


# In[ ]:


global_training_complete_df.to_json(HOME_TT + "/" + "combined_data/" + 'validation_split_10_districts.json')


# In[ ]:





# In[ ]:


global_training_complete_df = []
global_testing_df = []
global_validation_df = []


# In[ ]:


for district in tqdm(districts):
  filename = file(district)
  df = pd.read_json(filename, orient = 'index')
  court_complexes = df.index
  district_df = pd.DataFrame()
  for i in court_complexes:
    df_complex = pd.DataFrame(df.loc[i]['processed']).T
    if len(df_complex) == 0 or 'segments' not in df_complex:
      continue
    df_complex = df_complex[df_complex['decision'] != "don't know"]
    if len(df_complex) == 0:
      continue
    df_complex = df_complex.dropna(subset=['segments'])
    if len(df_complex)!=0:
        df_complex['complex'] = i
        district_df = pd.concat((district_df, df_complex))


  train, test = train_test_split(district_df, test_size=0.2, random_state = 42)
  train, val = train_test_split(train, test_size=0.125, random_state = 42)
  # print(train['decision'].describe())
  # print(test['decision'].describe()) random split works no need to stratify as such.

  base_target = HOME_TT + "/" + district + "/"
  train.to_json( base_target + 'training_split.json' , orient = 'index')
  val.to_json( base_target + 'validation_split.json' , orient = 'index')
  test.to_json(  base_target + 'testing_split.json' , orient = 'index')
  
  train['district'] = district
  val['district'] = district
  test['district'] = district

  global_training_complete_df.append(train)
  global_testing_df.append(test)
  global_validation_df.append(val)


# In[ ]:


global_training_complete_df = pd.concat(global_training_complete_df)
global_testing_df = pd.concat(global_testing_df)
global_validation_df = pd.concat(global_validation_df)


# In[ ]:


global_validation_df


# In[ ]:


global_training_complete_df['case_number'] = global_training_complete_df.index
global_testing_df['case_number'] = global_testing_df.index
global_validation_df['case_number'] = global_validation_df.index


# In[ ]:


global_training_complete_df.reset_index(inplace=True)
global_testing_df.reset_index(inplace=True)
global_validation_df.reset_index(inplace=True)


# In[ ]:


global_testing_df.to_json(HOME_TT + "/" + "combined_data/" + 'test_split_alldistricts.json')
global_validation_df.to_json(HOME_TT + "/" + "combined_data/" + 'val_split_alldistrict.json')
global_training_complete_df.to_json(HOME_TT + "/" + "combined_data/" + 'train_split_alldistrict_bail.json')


# In[ ]:


global_training_complete_df['decision'].value_counts()


# In[ ]:


global_testing_df['decision'].value_counts()


# In[ ]:


global_training_complete_df['district'].value_counts()


# In[ ]:


len(global_training_complete_df)


# In[ ]:


len(global_testing_df)


# In[ ]:


len(global_validation_df)


# In[ ]:


len(global_training_reduced_df)

