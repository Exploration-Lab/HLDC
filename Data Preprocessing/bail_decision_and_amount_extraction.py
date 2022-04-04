#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
from multiprocessing.pool import ThreadPool as Pool


# In[3]:


districts = ['agra', 'aligarh', 'varanasi', 'lucknow', 'ghaziabad', 'ambedkar_nagar', 'bahraich', 'azamgarh',
             'allahabad', 'balrampur', 'auraiya', 'barabanki', 'banda', 'bagpat', 'bhadohi', 'ballia', 'bijnor',
             'basti', 'bareilly', 'bulandshahar', 'chitrakoot', 'deoria', 'budaun', 'etah', 'etawah', 'farrukhabad',
             'faizabad', 'fatehpur', 'firozabad', 'gautam_buddha_nagar', 'ghazipur', 'hapur', 'gonda', 'hardoi',
             'hamirpur_up', 'jaunpur', 'gorakhpur', 'jalaun', 'jyotiba_phule_nagar', 'jhansi', 'hathras', 'kanpur_dehat',
             'kanpur_nagar', 'kannauj', 'kanshiramnagar', 'kheri', 'kaushambi', 'kushinagar', 'lalitpur', 'maharajganj',
             'mainpuri', 'meerut', 'mahoba', 'mirzapur', 'mathura', 'moradabad', 'muzaffarnagar', 'pratapgarhdistrict', 'pilibhit',
             'rampur', 'raebareli', 'mau', 'saharanpur', 'sant_kabir_nagar', 'shahjahanpur', 'shravasti', 'siddharthnagar', 'sitapur', 'unnao',
             'sultanpur', 'sonbhadra']

assert len(districts) == 71

def file(district):
  path = "<location-of-file-for-district>"
  return path


# ## Dividing district into chunks for multiprocessing

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


# ## Bail Decision Code

# In[4]:


def check_with_improper_spacing(token, text):
  space_indices  = []
  for i, ch in enumerate(token):
    if ch == ' ':
      space_indices.append(i)
  
  all_tokens = []
  for i in range(1<<len(space_indices)):
    curr_token = token
    temp = 0
    for j in range(len(space_indices)):
      if (i>>j)&1:
        new_token = ''
        for idx, ch in enumerate(curr_token):
          if idx == space_indices[j] - temp:
            continue
          new_token += ch
        curr_token = new_token
        temp += 1
    all_tokens.append(curr_token)

  for token in all_tokens:
    if token in text:
      return True
  return False
  
granted_tokens = ['स्वीकार किया जाता', 'स्वीकार करते हुये', 'स्वीकार किये जाते', 'रिहा किए जाने का आदेश दिया जाता',
                  'स्वीकार किये जाने योग्य है', 'पर्याप्त आधार प्रतीत होता है', 'पर्याप्त आधार पाता हूँ', 'आधार पर्याप्त है',
                  'पर्याप्त आधार दर्शित होता', 'रिहा किये जाने का आदेश दिया जाता', 'रिहा किया जावे',
                  'रिहा किया जाए', 'मुक्त किया जाता', 'रिहा कर दिये जायें',
                  'रिहा किया जाता है', 'रिहा किया जाये', 'रिहा कर दिया जाये']

dismissed_tokens = ['निरस्त किया जाता', 'निरस्त किये जाते', 'निरस्त किए जाते', 'खण्डित किया जाता','खण्डित किये जाते', 'पर्याप्त आधार नहीं है',
                    'पर्याप्त आधार प्रतीत नहीं होता', 'खारिज किया जाता', 'अस्वीकार']a


# In[5]:


def decision(result):
  for token in dismissed_tokens:
    if check_with_improper_spacing(token, result):
      return 'dismissed'
  
  for token in granted_tokens:
    if check_with_improper_spacing(token, result):
      return 'granted'
  
  return "don't know"


# ## Bail Amount

# In[6]:


#@title
counting = [
"एक",
"दो",
"तीन",
"चार",
"पांच",
"छः",
"सात",
"आठ",
"नौ",
"दस",
"ग्यारह",
"बारह",
"तेरह",
"चौदह",
"पंद्रह",
"सोलह",
"सत्रह",
"अट्ठारह",
"उन्निस",
"बीस",
"इक्कीस",
"बाईस",
"तेईस",
"चौबीस",
"पच्चीस",
"छब्बीस",
"सत्ताईस",
"अट्ठाईस",
"उनतीस",
"तीस",
"इकतीस",
"बत्तीस",
"तैंतीस",
"चौंतीस",
"पैंतीस",
"छ्त्तीस",
"सैंतीस",
"अड़तीस",
"उनतालीस",
"चालीस",
"इकतालीस",
"बयालीस",
"तैंतालीस",
"चौंतालीस",
"पैंतालीस",
"छियालीस",
"सैंतालीस",
"अड़तालीस",
"उनचास",
"पचास",
"इक्याबन",
"बावन",
"तिरेपन",
"चौबन",
"पचपन",
"छप्पन",
"सत्तावन",
"अट्ठावन",
"उनसठ",
"साठ",
"इकसठ",
"बासठ",
"तिरसठ",
"चौंसठ",
"पैंसठ",
"छियासठ",
"सड़सठ",
"अड़सठ",
"उनहत्तर",
"सत्तर",
"इकहत्तर",
"बहत्तर",
"तिहत्तर",
"चौहत्तर",
"पचहत्तर",
"छिहत्तर",
"सतहत्तर",
"अठहत्तर",
"उनासी",
"अस्सी",
"इक्यासी",
"बयासी",
"तिरासी",
"चौरासी",
"पचासी",
"छियासी",
"सतासी",
"अठासी",
"नवासी",
"नब्बे",
"इक्यानबे",
"बानवे",
"तिरानवे",
"चौरानवे",
"पचानवे",
"छियानवे",
"सत्तानवे",
"अट्ठानवे",
"निन्यानवे",
]


# In[7]:


amount_map = {
    '5-5 हजार': 10000,
    'बीस-बीस हजार': 40000,
    'पचीस-पचीस हजार': 50000,
    'तीस-तीस हजार': 60000,
    'पैतिस-पैतिस हजार': 70000,
    'चालीस-चालीस हजार': 80000,
    'पचास-पचास हजार': 100000,
    'साठ-साठ हजार': 120000,
    'सत्तर-सत्तर हजार': 140000,
    'पिचहत्तर-पिचहत्तर हजार': 150000,
    'एक-एक लाख': 200000,
    'पचहत्तर हजार': 75000,
    '10 हजार': 10000,
    'बीस हजार': 20000,
    'पचीस हजार': 25000,
    'तीस हजार': 30000,
    'पैतिस हजार': 35000,
    'चालीस हजार': 40000,
    'पचास हजार': 50000,
    'साठ हजार': 60000,
    'सत्तर हजार': 70000,
    'पिचहत्तर हजार': 75000,
    'एक लाख': 100000,
    'पांच लाख': 500000,
}
x = 1
for c in counting:
  amount_map[f"{c} हजार"] = x*1000 
  amount_map[f"{c}-{c} हजार"] = 2*x*1000
  amount_map[f"{c} लाख"] = x*100000 
  amount_map[f"{c}-{c} लाख"] = 2*x*100000
  amount_map[f"{x} हजार"] = x*1000 
  amount_map[f"{x}-{x} हजार"] = 2*x*1000
  amount_map[f"{x} लाख"] = x*100000 
  amount_map[f"{x}-{x} लाख"] = 2*x*100000 
  x += 1
print(amount_map.keys())


# In[8]:


rupay = 'रूपये'
rupay1 = 'रुपये'
only = '/-'
only1 = '/ -'
hazar = 'हजार'
lakh = 'लाख'
def get_amount(text):
  try:
    if only in text or only1 in text:
      def from_position(pos, text):
        while not text[pos].isnumeric():
          pos -= 1
        s = ''
        while pos > 0 and text[pos].isnumeric() or text[pos] == ',':
          s += text[pos]
          pos -= 1 
        s = s[::-1]
        s = int(s.replace(',', ''))
        return s

      positions = [i for i in range(len(text)) if text.startswith(only, i)]
      positions += [i for i in range(len(text)) if text.startswith(only1, i)]
      amount = 0
      for pos in positions:
        amount += from_position(pos, text)
      if amount < 1000:
        if hazar in text:
          return amount * 1000
        else:
          return amount * 100000
      return amount

    elif rupay in text or rupay1 in text:
      for word, value in amount_map.items():
        if check_with_improper_spacing(word, text):
          return value
      return -1
    else:
      return -1
  except:
    return -1


# ## Final 

# In[ ]:


def work(district):
  print(f"Enter: {district}")
  with open(file(district), 'r') as f:
    data = json.load(f)
  
  for court in data.keys():
    for case in data[court]['processed'].keys():
      temp = data[court]['processed'][case]
      temp['decision'] = decision(temp['result'])
      temp['bail_amount'] = get_amount(temp['result'])
  
  with open(file(district), 'w') as f:
    json.dump(data, f) 
  print(f"Exit: {district}")


# In[ ]:


from multiprocessing import Pool
import tqdm
with Pool(5) as p:
  for _ in tqdm.tqdm(p.map(work, district_chunks[0]), total=len(district_chunks[0])):
    pass

