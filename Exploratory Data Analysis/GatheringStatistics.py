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


import glob


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
  return f"../raw_pdfs/{district}/*"
  # complex_name = court_complexes[district]
  # l = []
  # for i in complex_name:
  #   s = f"../raw_pdfs/{district}/{i}/*.csv"
  #   l.append(s)
  # return l


# In[ ]:


court_complexes = {}
def district_court_complexes(district):
  complexes = glob.glob(file(district))
  complexes = [complex.split("/")[-1] for complex in complexes]
  court_complexes[district] = complexes


# In[ ]:


court_complexes


# In[ ]:


district_wise_case_types = {}


# In[ ]:


total = 0
def district_complexes_counts(district):
  metadata_csvs = []
  for i in file(district):
    metadata_csvs.extend(glob.glob(i))
  district_wise_case_types[district] = {}

  for csv in metadata_csvs:
    complex_ = csv.split("/")[-2]
    if complex_ not in district_wise_case_types[district]:
      district_wise_case_types[district][complex_] = {}
    df = pd.read_csv(csv)
    # print(complex_)
    for i in df['Case Type/Case Number/Case Year']:
      # total += 1
      vals = i.split("/")
      case_type = "/".join(vals[:-2])
      case_type = case_type.lower()
      year = vals[-1]
      if (case_type, year) in district_wise_case_types[district][complex_]:
        # print((case_type, year))
        district_wise_case_types[district][complex_][(case_type, year)] += 1
        # print(district_wise_case_types[district][complex_])
      else:
        # print((case_type, year))
        district_wise_case_types[district][complex_][(case_type, year)] = 1
      # district_wise_case_types[district][complex_][(case_type, year)] = district_wise_case_types[district][complex_].get((case_type, year), 0) + 1
  # print(total)


# In[ ]:


cnt = 0

for i in court_complexes:
  cnt += len(court_complexes[i])


# In[ ]:


cnt


# In[ ]:


from copy import deepcopy
district_chunks = []
# SET CHUNK SIZE
chunk_size = 71
len1, len2 = len(districts), 0
for i in range(0, len(districts), chunk_size):
  district_chunks.append(districts[i:min(len(districts), i + chunk_size)])
  len2 += len(district_chunks[-1])
  print(district_chunks[-1])
assert len2 == len1


# In[ ]:


for i in tqdm(districts):
  # print(i)
  district_court_complexes(i)


# In[ ]:


for i in tqdm(districts):
  # print(i)
  district_complexes_counts(i)


# In[ ]:


district_wise_case_types


# In[ ]:


district_wise_case_types['ghaziabad']


# In[ ]:


import re
import string


# In[ ]:


re.sub('rev$', "revision", 'civil rev ')


# In[ ]:


regex = re.compile('[%s]' % re.escape(string.punctuation))
def normalise(case_type):
  case_type = regex.sub(" ", case_type)
  case_type = re.sub('\d', " ", case_type)
  case_type = re.sub('xx+', "", case_type)
  case_type = " ".join(case_type.split())
  case_type = re.sub('trail', "trial", case_type)
  case_type = re.sub('reivision', "revision", case_type)
  case_type = re.sub('motar', "motor", case_type)
  case_type = re.sub('marrige', "marriage", case_type)
  case_type = re.sub('cri ', "criminal ", case_type)
  case_type = re.sub('spl', "special ", case_type)
  case_type = re.sub('xoriginal ', "original ", case_type)
  case_type = re.sub('xmisc ', "misc ", case_type)
  case_type = re.sub('xcriminal ', "criminal ", case_type)
  case_type = re.sub('xcomplaint ', "complaint ", case_type)
  case_type = re.sub('execuition ', "execution ", case_type)
  case_type = re.sub('crl ', "criminal ", case_type)
  case_type = re.sub('civ ', "civil ", case_type)
  case_type = re.sub('rev$', "revision", case_type)
  case_type = re.sub('revis$', "revision", case_type)
  case_type = re.sub('panchayt', "panchayat", case_type)
  case_type = re.sub('panch', "panchayat", case_type)
  case_type = re.sub('cr ', "criminal ", case_type)
  case_type = re.sub('xexecution', "execution", case_type)
  case_type = re.sub('apl', "appeal", case_type)
  case_type = re.sub('app$', "application", case_type)
  case_type = re.sub('summery', "summary", case_type)
  case_type = re.sub('special\s+t ', "special trial", case_type)
  case_type = re.sub('moter', "motor", case_type)
  case_type = re.sub('pet$', "petition", case_type)
  case_type = re.sub('dom viol', "domestic violence", case_type)
  # case_type = re.sub('summery', "summary", case_type)
  # case_type = re.sub('special\s+t ', "special trial", case_type)
  # case_type = re.sub('moter', "motor", case_type)
  # case_type = re.sub('pet$', "petition", case_type)
  case_type = re.sub('old', "", case_type)

  tmp = "".join(case_type.split())
  case_type = " ".join(case_type.split())
  if tmp.startswith("bail"):
    return "bail application"
  elif tmp.startswith('anticipatorybail'):
    return 'anticipatory bail'
  elif tmp.startswith('civil'):
    return 'civil cases'
  elif tmp.startswith('special'):
    return 'special cases'
  elif tmp.startswith('civilappeal'):
    return 'civil appeal'
  elif tmp.startswith('criminal'):
    return 'criminal cases'
  elif tmp.startswith('civilmisc') or tmp.startswith('misccivil'):
    return 'civil misc'
  elif tmp.startswith('warrantorsummonscri'):
    return 'warrant or summons criminal case'
  elif tmp.startswith('arb'):
    return 'arbitration'
  elif tmp.startswith('civilrevision'):
    return 'civil revision'
  elif tmp.startswith('motoraccidentclaim'):
    return 'motar accident claim'
  elif tmp.startswith('misccases'):
    return 'misc cases'
  elif tmp.startswith('misccriminal') or tmp.startswith("criminalmisc"):
    return 'misc criminal'
  elif tmp.startswith('matrimonial'):
    return 'matrimonial cases'
  elif tmp.startswith('criminalappeal'):
    return 'criminal appeal'
  elif tmp.startswith('originalcivilsuit'):
    return 'original civil suit'
  elif tmp.startswith('regular'):
    return 'regular suit'
  elif tmp.startswith('rent'):
    return 'rent case'
  elif tmp.startswith('finalreport'):
    return 'final report'
  elif tmp.startswith('sessiontrial'):
    return 'session trial'
  elif tmp.startswith('specialtrial'):
    return 'special trial'
  elif tmp.startswith('transferapplication'):
    return 'transfer application'
  elif tmp.startswith('scc'):
    return 'scc case'
  elif tmp.startswith('ndps'):
    return 'ndps case'
  elif tmp.startswith('sst'):
    return 'sst case'
  elif tmp.startswith('comp'):
    return 'complaint case'
  elif tmp.startswith('execution'):
    return 'execution case'
  elif tmp.startswith('landacq'):
    return 'land acquisition'
  elif tmp.startswith('summon'):
    return 'summon trial'
  elif tmp.startswith('arms'):
    return 'arms act'
  elif tmp.startswith('original'):
    return 'original suit'
  elif tmp.startswith('gaurdian'):
    return 'gaurdians and wards act'
  elif tmp.startswith('panchayat'):
    return 'panchayt revision'
  elif tmp.startswith('specialcase'):
    return 'special case'
  elif tmp.startswith('specialcriminal'):
    return 'special criminal'
  elif tmp.startswith('motorvehicleact'):
    return 'motor vehicle act'
  elif tmp.startswith('smallcausecourt'):
    return 'small cause court'
  elif tmp.startswith('regmisc'):
    return 'reg misc'
  elif tmp.startswith('specialsessiont') or tmp.startswith('specialsessionst'):
    return 'special session trials'
  elif tmp.startswith('juvenile'):
    return 'juvenile case'
  elif tmp.startswith('session'):
    return 'session trial'
  elif tmp.startswith('caveat'):
    return 'caveat application'
  elif tmp.startswith('domestic'):
    return 'domestic violence'
  elif tmp.startswith('upub'):
    return 'upub cases'
  elif tmp.startswith('reg'):
    return 'regular case'

  return case_type


# In[ ]:


overall_case_types = {}
s = 0
for i in district_wise_case_types:
  
  for j in district_wise_case_types[i]:
    for k in district_wise_case_types[i][j]:
      case_type = normalise(k[0])
      overall_case_types[case_type] = overall_case_types.get(case_type, 0) + district_wise_case_types[i][j][k]
      s += district_wise_case_types[i][j][k]
  # print(i, s)


# In[ ]:


overall_case_types


# In[ ]:


c = 0
for i in overall_case_types:
  c += overall_case_types[i]

print(c)


# In[ ]:


len(overall_case_types)


# In[ ]:


vals = overall_case_types.items()


# In[ ]:


sorted(list(vals), key = lambda x: x[1], reverse = True)


# In[ ]:


district_bail = {}

for i in district_wise_case_types:
  s = 0
  for j in district_wise_case_types[i]:
    for k in district_wise_case_types[i][j]:
      case_type = normalise(k[0])
      if case_type == "bail application":
        district_bail[i] = district_bail.get(i, 0) + district_wise_case_types[i][j][k]
      # overall_case_types[case_type] = overall_case_types.get(case_type, 0) + district_wise_case_types[i][j][k]
      s += district_wise_case_types[i][j][k]
  # district_case[i] = s


# In[ ]:


district_bail


# In[ ]:


import pandas as pd
district_case = pd.DataFrame(district_case.items())
district_case.columns = ['district', 'total_cases']


# In[ ]:


district_case['ratio'] = district_case[['bail_cases', 'total_cases']].apply(lambda x: x[0]/x[1], axis=1)


# In[ ]:


district_case['bail_cases'] = district_case['district'].apply(lambda x: district_bail[x])


# In[ ]:


district_case.to_csv("case-dist-across-districts.csv")

