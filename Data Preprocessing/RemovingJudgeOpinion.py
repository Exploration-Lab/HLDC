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


# # Paragraph Annotation

# In[ ]:


split_text_1 = ['विद्वान अधिवक्ता', 'विद्वान अधिवक्त', 'उभय पक्ष के तर्कों', 'विद्धान अधिवक्ता', 'उभय पक्ष के तकों', 'उभय पक्षों', 
                'उभय पक्षो', 'केस डायरी', 'सुना', 'दर्ज बयान', 'सुना', "उभय पक्ष के तर्को", "संलग्नप्रपत्रों", "उभय पक्ष", "प्रथम सूचना रिपोर्ट, केस डायरी",
                "प्रथम सूचना रिपोर्ट, पुलिस प्रपत्र"] 
split_text_2 = ['अवलोकन किया', 'परिशीलन किया', "अवलोकन के उपरान्त", "परीशिलन किया", "परिशीलन कर लिया है", "के अवलोकन से स्पष्ट"]

terms_to_look_for = {
    "s_judge": [
          split_text_1, split_text_2  # same sentence
    ],
    "judge_part_2": [
      [ "उभय पक्ष की बहस सुनने", 
       "पत्रावली के अवलोकन", "प्रपत्रों के अवलोकन से", "केस डायरी के अवलोकन से विदित", 
       "प्रपत्रो के अवलोकन से विदित", "केस डायरी के अवलोकन से स्पष्ट", "प्रपत्रों का सम्यक परिशीलन किया", "पत्रावली का अवलोकन करने से स्पष्ट",
       "जमानत के प्रकम पर साक्ष्य का विश्लेषण", "जमानत के प्रक्रम पर साक्ष्य का विश्लेषण", "न्यायालय द्वारा मामले के समस्त तथ्यों एवं परिस्थितियों", 
       "अभियोजन प्रपत्रों का अवलोकन किया गया", "इस स्तर पर गुण-दोष के आधार पर तथ्यों का मूल्यांकन करना", "केस डायरी व प्रथम सूचना रिपोर्ट के अवलोकन से स्पष्ट",
       "अभियोजन प्रपत्रं के अवलोकन से", "अभियोजन प्रपत्रों के अवलावेकन से स्पष्ट होता है", "तर्कों को सुना", "केस डायरी व अभियोजन प्रपत्रों के सम्यक रूपेण परिशीलन",
       "उपरोक्त तथ्य, परिस्थितियों तथा प्राथमिकी का अवलोकन करने से", "उपरोक्त समस्त तथ्य एवं परिस्थितियों को दृष्टिगत", "जमानतप्रार्थना पत्र पर सुना गया", 
       "पत्रावली के परिशीलन से स्पष्ट है", "विवेचना के दौरान दिनांक", "पत्रावली पर उपलब्ध पुलिस प्रपत्र व अन्य प्रलेखों के अवलोकनसे", "पत्रावली का अवलोकन किया",
       "मामले के तथ्यों व परिस्थितियों में पूरी तरह से स्पष्ट है", "केस डायरी में उपलब्ध साक्ष्य के अनुसार"
      ]            
    ],
    "facts by prosecutor": [
            [ "अभियोजन कथन इस प्रकार है", "अभियोजन का कथन इस प्रकार है", "अभियोजन कथानक के अनुसार", "अभियोजन कथानक इस प्रकार है",
             "अभियोजन कथानक यह है कि वादी", "अभियोजन के अनुसार पुलिस द्वारा दिनांक", "संक्षेप में अभियोजन केस के अनुसार", "संक्षेप में अभियोजन कथानक इस प्रकार है", 
             "अभियोजन के अनुसार प्रार्थी", "अभियोजन केस के अनुसार वादी", "अभियोजन केस के अनुसार", "संक्षेप में अभियोजन के अनुसार कथन",
             "अभियोजन कथानक संक्षेप में इस प्रकार है", "अभियोजन के अनुसार दिनांक", "अभियोजन पक्ष के कथनानुसार", "अभियोजन के अनुसार अभियुक्त", 
             "अभियोजन के अनुसार वादी मुकदमा"
            ]
    ],
    "public prosecutor": [
            ["जमानत का विरोध करते हुये अभियोजन की ओर से तर्क दिया गया है", 
             "विरोध मे अभियोजन का तर्क", "जमानत प्रार्थनापत्र के विरूद्ध आपत्ति", 
             "जमानत का घोर विरोध",  "अभियोजन द्वारा जमानत प्रार्थना पत्र का विरोध किया गया है", "जमानत प्रार्थना पत्रका विरोध करते हुए कहा",
             "विद्वान सहायक जिला शासकीय अधिवक्ता (फौजदारी) द्वारा जमानत प्रार्थना पत्र काविरोध किया गया है", 
             "अभियोजन की तरफ से सहायक जिला शासकीय अधिवक्ता, फौजदारी द्वारा जमानतप्रार्थना पत्र का इस आधार पर विरोध किया गया", 
             "विद्वान जिला शासकीय अधिवक्ता (फौजदारी)द्वारा प्रार्थनापत्र काविरोध करते हुए कथन किया है कि", 
             "जमानत प्रार्थना पत्र का विरोध", "प्रार्थनापत्र का विरोध",
             "विद्वान अभियोजन अधिकारी द्वारा जमानत का विरोध", "जमानत प्रार्थना-पत्रका विरोध", "विद्वान जिला शासकीय अधिवक्ता (फौजदारी)द्वारा विरोध करते हुए तर्क", 
             "प्रार्थनापत्र का घोर विरोध", "जमानत प्रार्थनापत्र काकड़ा विरोध", "जमानतका विरोध करते हुए तर्क दिया गया", 
             "जमानत का विरोध करते हुए कथनकिया गया कि", "जमानत का विरोध", "जमानत प्रा० पत्र का विरोध", "जमानत प्रार्थना-पत्र काविरोध"
             ]
    ],
    "defendant": [
            ["अभियुक्त के विद्वान अधिवक्ता का तर्क है", "अभियुक्त की ओर से विद्वान अधिवक्ता द्वारा जमानत प्रार्थना पत्र पर बल देते हुए", 
             "अभियुक्त के विद्वान अधिवक्ता द्वारा जमानत प्रार्थना पत्र में यह आधार", "अभियुक्त निर्दोष है", 
             "अभियुक्त के विद्वान अधिवक्ता द्वारा तर्क प्रस्तुत किए गए कि", "अभियुक्त निर्दोष है", "बचाव पक्ष द्वारा यह आधार लिया गया", 
             "अभियुक्तगण की तरफ से जमानत प्रार्थना पत्र में कथन किया गया है", "विद्वान अधिवत्ता प्रार्थी /अभियुक्त ने तथ्य एवं विधि के तर्को", 
             "में झूठा एवं रंजिशन फंसाया गया", "में झूंठा व रंजिशन फंसाया गया", "विद्वान अधिवक्ता द्वारा निम्नलिखित तर्क किये गये", 
             "अभियुक्त की ओर से विद्वान अधिवक्ता द्वारा तर्क प्रस्तुत किया गया है", "अभियुक्तनिर्दोष है", 
             "अभियुक्त की तरफ से जमानत प्रार्थना पत्र में कथन किया गया है", "अभियुक्त निर्दोषहै", "अभियुक्त ने तर्क प्रस्तुत करते हुए कहा है कि", 
             "अभियुक्तगण की ओर से विद्वान अधिवक्तागण द्वारा जमानत प्रार्थना पत्र पर बल देते हुए तर्क प्रस्तुत किये गये",
             "जमानत प्रार्थना पत्रो के समर्थन में", "अभियुक्त द्वारा जमानत पर अवमुक्त किये जाने के लिए", 
             "अभियुक्त की ओर से उसके विद्वान अधिवक्ता द्वारा जमानत प्रार्थनापत्र पर बहसकरते हुए", 
             "अभियुक्तगण की ओर से जमानत प्रार्थनापत्र में मुख्य आधार यह लिए गये हैं", "अभियुक्तपर लगाया गया आरोप पूर्णतया असत्य व निराधार है", 
             "अभियुक्त के विद्वान अधिवक्ता की ओर से तर्क दिया गया कि", "अभियुक्तागण की ओर से कथन किया गया है कि", 
             "अभियुक्त के विद्वान अधिवक्ता की तरफ से मुख्य तर्क यह दिये गये है", "अभियुक्तगण की ओर से विद्वान अधिवक्ता द्वारा जमानत प्रार्थना पत्र पर बल देते हुए",
             "अभियुक्ता की ओर से यह आधार लिया गया है कि", "अभियुक्तानिर्दोष है", "अभियुक्तगण की ओर से यह तर्क दिया गया है", 
             "अभियुक्त के विद्वान अधिवक्ता द्वारा जमानत प्रार्थना-पत्र पर बहस करते हुए", "अभियुक्त को उक्त केस में झूठा फॅसाया गयाहै", \
             "बचाव पक्ष के अधिवक्ता द्वारा तर्क प्रस्तुत किया गया", "अभियुक्त द्वारा जमानत पर अवमुक्त किये जाने के लिए अपने अनुरोधके समर्थन में", 
             "अभियुक्तगण की ओर से बहस करते हुऐ उसके विद्वान अधिवक्ता द्वारा तर्क दियागया"
             ]
    ],
    # "p_initial declaration": [
    #     ["प्रस्तुत प्रतिभू आवेदन", "प्रार्थना पत्र के समर्थन में", "प्रथम जमानत प्रार्थना-पत्र", "प्रथम जमानत प्रार्थनापत्र", "जमानत प्रार्थना पत्र", "जमानत प्रार्थना-पत्र"], 
    #     ["किसी अन्य न्यायालय में", "अन्य किसी न्यायालय में", "किसी न्यायालय", "अन्यन्यायालय", "अन्य न्यायालय"]         #same para
    # ]
}


# In[ ]:


names =[]
import os
files = os.listdir("./ner_data/ner_data/")
for file_ in files:
  with open(f"./ner_data/ner_data/{file_}") as f:
    import json
    names+=json.load(f)
import re
names = set(names)
names.remove("किया")
names.remove("प्रार्थना")
names.remove("गया")
names.remove("लिया")
def remove_NER(text):
  text = re.sub(r'((\+*)((0[ -]*)*|((91 )*))((\d{12})+|(\d{10})+))|\d{5}([- ]*)\d{6}', '<फ़ोन-नंबर>', text)
  text = re.sub(r'((\+*)((०[ -]*)*|((९१ )*))((\d{१2})+|(\d{१०})+))|\d{५}([- ]*)\d{६}', '<फ़ोन-नंबर>', text)
  text = text.split()
  punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
  for i,it in enumerate(text):
    for p in punc:
      it = it.replace(p,"")
    #print(it)
    if it in names:
      text[i]="<नाम>"
  return " ".join(text)


# In[ ]:


regex_arrays = [split_text_1, split_text_2, terms_to_look_for['judge_part_2'][0], terms_to_look_for["facts by prosecutor"][0],
                terms_to_look_for["public prosecutor"][0], terms_to_look_for["defendant"][0]]

for i in regex_arrays:
  for j in  range(len(i)):
    i.append(remove_NER(i[j]))


# In[ ]:




# In[ ]:


import re

def strip_spaces(string):
    no_spaces = ''

    for pos, char in enumerate(string):
        if re.match(r'\S', char):  # upper \S matches non-whitespace chars
            no_spaces += char
    return no_spaces

def compile_regex(list_of_terms):
  expression = re.compile(
        '(' + 
        '|'.join(re.escape(strip_spaces(item)) for item in list_of_terms) +
        ')')
  
  return expression

def check_for_presence_of_any_one(content, expression):
    if not expression.search(content):
        return False
    return True

terms_to_look_for_regex = {
}

for i in terms_to_look_for:
  r_list = []
  for j in terms_to_look_for[i]:
    r_list.append(compile_regex(j))
  terms_to_look_for_regex[i] = r_list


def find_labels(paras):
  para_temp = []

  ## For para level
  # for i in paras:
  #   para_temp.append(strip_spaces(i))
  
  # ## For sentence level
  for i in paras:
    para_temp.extend(list(map(strip_spaces, re.split("।|\|", i))))

  paras = para_temp

  para_labels = [None]*len(paras)

  starter = 0

  for i in paras[:]:
    
    for term in terms_to_look_for_regex:
      if term.startswith("p_"):
        results = []
        for j in terms_to_look_for_regex[term]:
          results.append(check_for_presence_of_any_one(i, j))
        if all(results):
          if para_labels[starter] is None:
            para_labels[starter] = {term}
          else:
            para_labels[starter].add(term)

      elif term.startswith("s_"):
        sentencewise = re.split("।|\|", i)

        for sentence in sentencewise:
          results = []
          for j in terms_to_look_for_regex[term]:
            results.append(check_for_presence_of_any_one(i, j))
          if all(results):
            if para_labels[starter] is None:
              para_labels[starter] = {term}
            else:
              para_labels[starter].add(term)
      else:
        for j in terms_to_look_for_regex[term]:
          if check_for_presence_of_any_one(i, j):
            if para_labels[starter] is None:
              para_labels[starter] = {term}
            else:
              para_labels[starter].add(term)

    starter += 1

  return para_labels


# In[ ]:


# annotating every sentence
def get_sentences_of_para(para):
  para_temp = []
  para_temp.extend(re.split("।|\|", para))
  
  return para_temp

def annotate_all(paras):
  paras, labels_found = paras[0], paras[1]
  sentence_counter = 0
  current_label = None
  # print(labels_found, len(labels_found))
  for para in paras:
     current_label = None
     sentences = get_sentences_of_para(para)
    #  print(sentences)

     for i in sentences:
      #  print(sentence_counter, i)
       present = labels_found[sentence_counter]
       if present is not None and 'facts by prosecutor' in present and 'public prosecutor' in present:
         present = {'public prosecutor'}

       if current_label is not None:
         if present is None:
           labels_found[sentence_counter] = current_label
         else:
           if present is not None and len(present) == 1:
             current_label = present
             labels_found[sentence_counter] = current_label
       else:
         if present is not None and len(present) == 1:
             current_label = present
             labels_found[sentence_counter] = current_label

       sentence_counter += 1

  # print(labels_found)


# In[ ]:


total_count = 0
def analyse(r, labels_found, case):
    global total_count
    # r['labels'] = labels_found
    not_after_wards = 0
    total_idx = 0
    judge_opinion = None
    arguments = None
    num_tokens = 0
    total_idx += 1
    labels = set()

    s1 = False
    s2 = False
    done = False
    ptr = len(labels_found) - 1
    for s3 in labels_found[::-1]:
      if done:
        break
      if s3 is not None:
        for s in s3:
          if s == 's_judge' or s == 'judge_part_2':
            s1 = True
            done = True
            break

          if (s == 'public prosecutor' or s == 'defendant' or s == 'facts by prosecutor'):
            s2 = True

      elif not done:
        # print("Assigning")
        labels_found[ptr] = {'s_judge'}
        ptr -= 1

    if s1 and not s2:
      not_after_wards += 1
      judge_opinion = []
      arguments = []

      cnt = 0

      for i in r['body']:
        sentences = get_sentences_of_para(i)
        args = ""
        opinions = ""
        set_ = False
        for j in sentences:
          if j.isspace() or len(j) == 0:
            cnt += 1
            continue
          if labels_found[cnt] is not None and ("s_judge" in labels_found[cnt] or "judge_part_2" in labels_found[cnt]):
            opinions += j + "। "
            set_ = True
          elif labels_found[cnt] is None and set_:
            opinions += j + "। "
          else:
            args += j + "। "
            set_ = False
          cnt += 1

        if len(args) != 0:
          arguments.append(args)
        if len(opinions) != 0:
          judge_opinion.append(opinions)

      if judge_opinion is not None and len(judge_opinion) != 0 and len(arguments) != 0:
        # print(case)
        total_count += 1
        r['segments'] ={
          'judge-opinion': judge_opinion,
          'facts-and-arguments': arguments
        }


# In[ ]:





# # Worker code

# In[ ]:


def work(district):
#   global total_count
# district = 'agra'
  print(f"Enter: {district}")
  with open(file(district), 'r') as f:
    data = json.load(f)

  for court in data.keys():
    df = pd.DataFrame(data[court]['processed']).T
    
    final_df = df
    if len(final_df) == 0:
      continue
    final_df['labels_found'] = final_df['body'].apply(find_labels)
    final_df[['body', 'labels_found']].apply(annotate_all, axis = 1)

    for case in data[court]['processed'].keys():
      temp = data[court]['processed'][case]
      analyse(temp, final_df.loc[case]['labels_found'], case)
  # print(total_count)

  with open(file(district), 'w') as f:
    json.dump(data, f)  
  print(f"Exit: {district}")



import numpy as np
indexes = []
correct = []
court = list(data.keys())[0]
for idx in data[court]['processed']:
  if 'segments' not in data[court]['processed'][idx]:
    if len(data[court]['processed'][idx]['body']) >= 4:
    #   distinct = set()
      # for i in data[court]['processed'][idx]['labels']:
    #     if i is None:
    #       continue
    #     for j in i:
    #       distinct.add(j)
    #   if len(distinct) > 1:
        indexes.append(idx)
  # else:
  #   correct.append(idx)


# In[ ]:


indexes


# In[ ]:



def work_count(district):
  print(f"Enter: {district}")
  filename = file(district)
  df = pd.read_json(filename, orient = 'index')
  for i in df.index:
      # print(i)
      df1 = pd.DataFrame(df['processed'][i])
      df1 = df1.T
      if len(df1) == 0:
        continue
      # final_df = df1[df1['decision'] != "don't know"]
      final_df = df1
      total = len(final_df)
      if 'segments' in final_df.columns:
        final_df = final_df.dropna(subset=['segments'])
        processed = len(final_df)
        print(district, ",", i, ",", processed, ",", total)
      else:
        print(district, ",", i, ",", 0, ",", total)

  print(f"Exit: {district}")


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


aba


# In[ ]:


from multiprocessing import Pool
import tqdm
all_districts_stats = []
for dst in tqdm.tqdm(districts):
  val = work_count(dst)
  all_districts_stats.append(val)

# with Pool(20) as p:
#   for _ in tqdm.tqdm(p.map(work, district_chunks[0]), total=len(district_chunks[0])):
#     pass


# In[ ]:


stats = pd.DataFrame(val)


# In[ ]:


val


# In[ ]:


stats.to_csv("districtwise_judge_opinion.csv")


# In[ ]:


total_count

