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
get_ipython().run_line_magic('load_ext', 'autotime')


# In[ ]:


import multiprocessing

cores = multiprocessing.cpu_count()
print(cores)


# In[ ]:


HOME = "path to folder/all_cases_pickles/"


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


ignore = ['ocr_documents', 'raw_documents', 'raw_pdfs']
new_files = []
for district in glob.glob(HOME + "*"):
        if district.split('/')[-1] not in districts: 
            continue 
        for court in glob.glob(district + '/*'):
            take = True
            for ig in ignore:
                if ig in court:
                    take = False
                    break
            if not take:
                continue
            for fname in glob.glob(court + "path to folder/*all_bail_cases.pickle", recursive=True):
                new_files.append(fname)
district_files = {}
for f in new_files:
    components = f.split('/')
    district = components[-3]
    court = components[-2]
    if district not in district_files:
        district_files[district] = {}
    district_files[district][court] = f
print(json.dumps(district_files, indent = 2))


# In[ ]:


from copy import deepcopy
districts = deepcopy(list(district_files.keys()))
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


stopwords = [
  "मैं", "मुझको", "मेरा", "अपने आप को", "हमने", "हमारा", "अपना", "हम", "आप", "आपका", "तुम्हारा", "अपने आप", "स्वयं", "वह", "इसे", 
"उसके", "खुद को", "कि वह", "उसकी", "उसका", "खुद ही", "यह", "इसके", "उन्होने", "अपने", "क्या", "जो", "किसे", "किसको", "कि", "ये",
 "हूँ", "होता है", "रहे", "थी", "थे", "होना", "गया", "किया जा रहा है", "किया है", "है", "पडा", "होने", "करना", "करता है", "किया", "रही", 
 "एक", "लेकिन", "अगर", "या", "क्यूंकि", "जैसा", "जब तक", "जबकि", "की", "पर", "द्वारा", "के लिए", "साथ", "के बारे में", "खिलाफ", 
 "बीच", "में", "के माध्यम से", "दौरान", "से पहले", "के बाद", "ऊपर", "नीचे", "को", "से", "तक", "से नीचे", "करने में", "निकल", "बंद", "से अधिक",
  "तहत", "दुबारा", "आगे", "फिर", "एक बार", "यहाँ", "वहाँ", "कब", "कहाँ", "क्यों", "कैसे", "सारे", "किसी", "दोनो", "प्रत्येक", 
 "ज्यादा", "अधिकांश", "अन्य", "में कुछ", "ऐसा", "में कोई", "मात्र", "खुद", "समान", "इसलिए", "बहुत", "सकता", "जायेंगे", "जरा", "चाहिए", 
 "अभी", "और", "कर दिया", "रखें", "का", "हैं", "इस", "होता", "करने", "ने", "बनी", "तो", "ही", "हो", "इसका", "था", "हुआ", "वाले", 
 "बाद", "लिए", "सकते", "इसमें", "दो", "वे", "करते", "कहा", "वर्ग", "कई", "करें", "होती", "अपनी", "उनके", "यदि", "हुई", "जा", "कहते", 
 "जब", "होते", "कोई", "हुए", "व", "जैसे", "सभी", "करता", "उनकी", "तरह", "उस", "आदि", "इसकी", "उनका", "इसी", "पे", "तथा", "भी",
 "परंतु", "इन", "कम", "दूर", "पूरे", "गये", "तुम", "मै", "यहां", "हुये", "कभी", "अथवा", "गयी", "प्रति", "जाता", "इन्हें", "गई", "अब",
 "जिसमें", "लिया", "बड़ा", "जाती", "तब", "उसे", "जाते", "लेकर", "बड़े", "दूसरे", "जाने", "बाहर", "स्थान", "उन्हें ", "गए", "ऐसे", 
 "जिससे", "समय", "दोनों", "किए", "रहती", "इनके", "इनका", "इनकी", "सकती", "आज", "कल", "जिन्हें", "जिन्हों", "तिन्हें", "तिन्हों", 
 "किन्हों", "किन्हें", "इत्यादि", "इन्हों", "उन्हों", "बिलकुल", "निहायत", "इन्हीं", "उन्हीं", "जितना", "दूसरा", "कितना", "साबुत", "वग़ैरह", 
 "कौनसा", "लिये", "दिया", "जिसे", "तिसे", "काफ़ी", "पहले", "बाला", "मानो", "अंदर", "भीतर", "पूरा", "सारा", "उनको", "वहीं", "जहाँ",
 "जीधर", "के", "एवं", "कुछ", "कुल", "रहा", "जिस", "जिन", "तिस", "तिन", "कौन", "किस", "संग", "यही", "बही", "उसी", "मगर",
 "कर", "मे", "एस", "उन", "सो", "अत", 'के', 'है।', '/', 'था।', 
 

 # varanasi
 "न्यायालय", "सत्र", "न्यायाधीश", "जमानत", "प्रार्थना", "पत्र", "अभियुकतगण", "अभियुक्त", "जिला", "शासकीय", "तर्को", "नहीं", "आवेदक", "प्रार्थी", 
 "प्रार्थनापत्र", "विद्वान", "अधिवक्ता", "Bail", "Application", "वाराणसी", "प्रथम", "सं०ः", "तर्क", "अलावा", "अभियुक्तगण", "बनाम", "संख्या", "धारा",
 "बनाम", "राज्य", "दिन", "विवाद", "करती", "यानी", "भा०दं०्सं०", "आदेश", "दिनांक", "होकर", "जिसके", "हेतु", "तथ्यों", "कानूनी", "District", 
 "Court", "वालों", "लोगों", "मुकदमा", "पत्रावली", "करेंगे", 'प्रस्तुत', 'अपराध', 'अभियोजन', 'पुत्र', 'रिपोर्ट', 'सूचना', 'भादंसं', 'व्यक्ति', 
 'पंजीकृत', 'मामले', 'बताया', 'मुअसं', 'रूपये', 'पुलिस', 'करायी', 'अन्तर्गत', 'अधिनियम', 'उपरोक्त', 'अग्रिम', 'पीड़िता', 'कराया', 'अवलोकन', 'पूर्व', 
 'विशेष', 'आपराधिक', 'Location', 'Sessions', 'मजिस्ट्रेट', 'प्रदेश', 'उत्तर', 'निवासी', 'बरामद', 'मैंने', 'वादिनी', 'भादसं', 'बरामदगी', 'निरस्त',
 'जिलावाराणसी', 'संबंध', 'अंतर्गत', 'प्रकार', 'इतिहास', 'बरामदगी', 'सहायक', 'साक्षी', 'न्यायिक', 'माननीय', 'पाण्डेय', 'संक्षेप', 
 'फौजदारी', 'कथानक', 'Reason', 'Document', 'प्रपत्रों', 'उपलब्ध', 'अभियुक्ता', 'विवेचना', 'सम्बन्ध', 'विरूद्ध', 'विरोध', 'ग्राम', 'मुकदमें', 'sessssessssneeeess'

 # agra
 'कारित', 'मिलकर', 'याचना', 'श्रीमती', 'अभियोजक', 'अधिकारी', 'निरूद्ध', 'याचना', 'दिनाक', 'विपक्षी', 'निर्दोष', 'प्राथमिकी', 'State', 'सम्बन्धित',
  'आधारों', 'जायेगा', 'अभियोजक', 'आवेदन', 'अभियोजक', 'प्रदान', 'अपराधी'
]

stopwords_list2 = ['', 'अर्थात', 'कुछ', 'तेरी', 'साबुत', 'अपनि', 'हूं', 'काफि', 'यिह', 'जा' ,'दे', 'देकर' ,'रह', 'कह' , 
                   'कर' , 'कहा', 'बात' , 'जिन्हों', 'किर', 'कोई', 'हे', 'कोन', 'रहा', 'सब', 'सो', 'तक', 'इंहें', 'इसकि', 
                   'अपनी', 'दबारा', 'सभि', 'होते', 'भीतर', 'निचे', 'घर', 'उन्हें', 'उन्ह' , 'मेरे' , 'था', 'व', 'इसमें', 'उसी', 
                   'बिलकुल', 'होति', 'गया', 'सकता', 'अपना', 'लिये', 'उसका', 'पर', 'दवारा', 'गए', 'है', 'कितना', 'भि', 'लिए', 'वुह ',
                   'ना', 'किसि', 'परन्तु', 'किन्हें', 'बहुत', 'भी', 'तुम्हारे', 'निहायत', 'उन्हीं', 'वहिं', 'हैं', 'उन्हों', 'इतयादि','यहाँ', 'तब', 'पूरा', 
                   'क्योंकि', 'कौनसा', 'आप', 'हुअ', 'ऐसे', 'एस', 'कारण', 'अप', 'पहले', 'तुम', 'जेसा', 'तिस', 'लेकिन', 'कहते', 'मगर', 'करता', 'संग', 
                   'सभी', 'जीधर', 'किंहों', 'हि', 'द्वारा', 'हुआ', 'तू', 'जिंहें', 'उसने', 'पास', 'वहां', 'वह', 'किंहें', 'इंहों', 'मुझ', 'कुल', 'तिंहों', 'का',
                   'मेरी', 'तेरे', 'उनके', 'क्या', 'जहाँ', 'काफ़ी', 'वर्ग', 'वरग','बही', 'ये', 'जिस', 'इसि', 'हुई', 'साम्हने', 'नहिं', 'जैसे', 'वहीं', 'दिया',
                   'अभी', 'यहि', 'वग़ैरह', 'उनकि', 'न', 'जा','बनि', 'हें', 'यिह ', 'उन', 'को', 'तिन्हों', 'उन्होंने', 'तुझे', 'उसे', 'होने', 'इन्हीं', 'थे',
                   'उंहिं', 'अपने', 'में', 'फिर','यही', 'नीचे', 'होती', 'तिसे', 'हम', 'यदि', 'सारा', 'कर', 'सकते', 'कोइ', 'और', 'जिंहों', 'तिंहें', 'दूसरे', 
                   'जब', 'रहे','अत', 'मानो', 'जिन', 'बाद', 'उनका', 'किया', 'या', 'उनकी', 'कौन', 'ऐसा', 'सबसे', 'अनुसार', 'दुसरे', 'इन', 'अदि','जिसे',
                   'उसकी', 'इत्यादि', 'करना', 'यहां', 'हुए', 'तेरा', 'आदि', 'पर  ', 'वाले', 'कहता', 'किन्हों', 'किसे', 'जिन्हें', 'मे','होता', 'करने', 'साभ',
                   'अभि', 'उसको', 'कई', 'बनी', 'के', 'इन्हें', 'वहाँ', 'कोनसा', 'कइ', 'इनका', 'थि', 'बाला','ऱ्वासा', 'हो', 'उंहें', 'दुसरा', 'वे', 'भितर',
                   'जेसे', 'एवं', 'अंदर', 'दो', 'साथ', 'करें', 'जिधर', 'तरह', 'उसि', 'इस', 'एसे', 'तिन', 'नहीं', 'से','न','उनको', 'किस', 'किसी', 'इसी',
                   'मैं', 'यह', 'हुइ', 'ले', 'कि', 'की', 'इसलिये', 'रवासा', 'ने', 'जैसा', 'वह ', 'तिन्हें', 'वुह', 'उस', 'उंहों', 'वगेरह', 'उसके', 'मुझे', 'करते',
                   'जितना', 'जहां', 'इन्हों', 'इसके', 'होना', 'इसका', 'इंहिं', 'एक', 'जो', 'पे', 'ही', 'तो', 'थी', 'रखें', 'इसे', 'इन ', 'के', 'बहि', 'पुरा', 
                   'ओर', 'इसकी']

stopwords_list3 = [
                   "अत", "अपना", "अपनी", "अपने", "अभी", "अंदर", "आदि", "आप", "इत्यादि", "इन ", "इनका", "इन्हीं", "इन्हें", "इन्हों", "इस", "इसका", 
"इसकी", "इसके", "इसमें", "इसी", "इसे", "उन", "उनका", "उनकी", "उनके", "उनको", "उन्हीं", "उन्हें", "उन्हों", "उस", "उसके", "उसी",
"उसे", "एक", "एवं", "एस", "ऐसे", "और", "कई", "कर", "करता", "करते", "करना", "करने", "करें", "कहते", "कहा", "का", "काफ़ी", "कि", 
"कितना", "किन्हें", "किन्हों", "किया", "किर", "किस", "किसी", "किसे", "की", "कुछ", "कुल", "के", "को", "कोई", "कौन", "कौनसा", "गया", 
"घर", "जब", "जहाँ", "जा", "जितना", "जिन", "जिन्हें", "जिन्हों", "जिस", "जिसे", "जीधर", "जैसा", "जैसे", "जो", "तक", "तब", "तरह", 
"तिन", "तिन्हें", "तिन्हों", "तिस", "तिसे", "तो", "था", "थी", "थे", "दबारा", "दिया", "दुसरा", "दूसरे", "दो", "द्वारा", "न", "नके", "नहीं", 
"ना", "निहायत", "नीचे", "ने", "पर", "पहले", "पूरा", "पे", "फिर", "बनी", "बही", "बहुत", "बाद", "बाला", "बिलकुल", "भी", "भीतर", "मगर", 
"मानो", "मे", "में", "यदि", "यह", "यहाँ", "यही", "या", "यिह", "ये", "रखें", "रहा", "रहे", "ऱ्वासा", "लिए", "लिये", "लेकिन", "व", "वग़ैरह", 
"वर्ग", "वह", "वहाँ", "वहीं", "वाले", "वुह", "वे", "सकता", "सकते", "सबसे", "सभी", "साथ", "साबुत", "साभ", "सारा", "से", "सो", "संग", 
"ही", "हुआ", "हुई", "हुए", "है", "हैं", "हो", "होता", "होती", "होते", "होना", "होने", "अंदर", "अत", "अदि", "अप", "अपना", "अपनि", 
"अपनी", "अपने", "अभि", "अभी", "आदि", "आप", "इंहिं", "इंहें", "इंहों", "इतयादि", "इत्यादि", "इन", "इनका", "इन्हीं", "इन्हें", "इन्हों", "इस", 
"इसका", "इसकि", "इसकी", "इसके", "इसमें", "इसि", "इसी", "इसे", "उंहिं", "उंहें", "उंहों", "उन", "उनका", "उनकि", "उनकी", "उनके", 
"उनको", "उन्हीं", "उन्हें", "उन्हों", "उस", "उसके", "उसि", "उसी", "उसे", "एक", "एवं", "एस", "एसे", "ऐसे", "ओर", "और", "कइ", "कई", 
"कर", "करता", "करते", "करना", "करने", "करें", "कहते", "कहा", "का", "काफि", "काफ़ी", "कि", "किंहें", "किंहों", "कितना", "किन्हें", 
"किन्हों", "किया", "किर", "किस", "किसि", "किसी", "किसे", "की", "कुछ", "कुल", "के", "को", "कोइ", "कोई", "कोन", "कोनसा", "कौन", 
"कौनसा", "गया", "घर", "जब", "जहाँ", "जहां", "जा", "जिंहें", "जिंहों", "जितना", "जिधर", "जिन", "जिन्हें", "जिन्हों", "जिस", "जिसे", "जीधर", 
"जेसा", "जेसे", "जैसा", "जैसे", "जो", "तक", "तब", "तरह", "तिंहें", "तिंहों", "तिन", "तिन्हें", "तिन्हों", "तिस", "तिसे", "तो", "था", "थि", 
"थी", "थे", "दबारा", "दवारा", "दिया", "दुसरा", "दुसरे", "दूसरे", "दो", "द्वारा", "न", "नहिं", "नहीं", "ना", "निचे", "निहायत", "नीचे", "ने", 
"पर", "पहले", "पुरा", "पूरा", "पे", "फिर", "बनि", "बनी", "बहि", "बही", "बहुत", "बाद", "बाला", "बिलकुल", "भि", "भितर", "भी", "भीतर", 
"मगर", "मानो", "मे", "में", "यदि", "यह", "यहाँ", "यहां", "यहि", "यही", "या", "यिह", "ये", "रखें", "रवासा", "रहा", "रहे", "ऱ्वासा", "लिए", 
"लिये", "लेकिन", "व", "वगेरह", "वरग", "वर्ग", "वह", "वहाँ", "वहां", "वहिं", "वहीं", "वाले", "वुह", "वे", "वग़ैरह", "संग", "सकता", "सकते", 
"सबसे", "सभि", "सभी", "साथ", "साबुत", "साभ", "सारा", "से", "सो", "हि", "ही", "हुअ", "हुआ", "हुइ", "हुई", "हुए", "हे", "हें", "है", 
"हैं", "हो", "होता", "होति", "होती", "होते", "होना", "होने", "मैं", "मुझको", "मेरा", "अपने आप को", "हमने", "हमारा", "अपना", "हम", "आप", 
"आपका", "तुम्हारा", "अपने आप", "स्वयं", "वह", "इसे", "उसके", "खुद को", "कि वह", "उसकी", "उसका", "खुद ही", "यह", "इसके", "उन्होने", 
"अपने", "क्या", "जो", "किसे", "किसको", "कि", "ये", "हूँ", "होता है", "रहे", "थी", "थे", "होना", "गया", "किया जा रहा है", "किया है", "है", 
"पडा", "होने", "करना", "करता है", "किया", "रही", "एक", "लेकिन", "अगर", "या", "क्यूंकि", "जैसा", "जब तक", "जबकि", "की", "पर", 
"द्वारा", "के लिए", "साथ", "के बारे में", "खिलाफ", "बीच", "में", "के माध्यम से", "दौरान", "से पहले", "के बाद", "ऊपर", "नीचे", "को", "से", 
"तक", "से नीचे", "करने में", "निकल", "बंद", "से अधिक", "तहत", "दुबारा", "आगे", "फिर", "एक बार", "यहाँ", "वहाँ", "कब", "कहाँ", "क्यों", 
"कैसे", "सारे", "किसी", "दोनो", "प्रत्येक", "ज्यादा", "अधिकांश", "अन्य", "में कुछ", "ऐसा", "में कोई", "मात्र", "खुद", "समान", "इसलिए", 
"बहुत", "सकता", "जायेंगे", "जरा", "चाहिए", "अभी", "और", "कर दिया", "रखें", "का", "हैं", "इस", "होता", "करने", "ने", "बनी", "तो", "ही", 
"हो", "इसका", "था", "हुआ", "वाले", "बाद", "लिए", "सकते", "इसमें", "दो", "वे", "करते", "कहा", "वर्ग", "कई", "करें", "होती", "अपनी", 
"उनके", "यदि", "हुई", "जा", "कहते", "जब", "होते", "कोई", "हुए", "व", "जैसे", "सभी", "करता", "उनकी", "तरह", "उस", "आदि", "इसकी", 
"उनका", "इसी", "पे", "तथा", "भी", "परंतु", "इन", "कम", "दूर", "पूरे", "गये", "तुम", "मै", "यहां", "हुये", "कभी", "अथवा", "गयी", "प्रति", 
"जाता", "इन्हें", "गई", "अब", "जिसमें", "लिया", "बड़ा", "जाती", "तब", "उसे", "जाते", "लेकर", "बड़े", "दूसरे", "जाने", "बाहर", "स्थान", 
"उन्हें ", "गए", "ऐसे", "जिससे", "समय", "दोनों", "किए", "रहती", "इनके", "इनका", "इनकी", "सकती", "आज", "कल", "जिन्हें", "जिन्हों", 
"तिन्हें", "तिन्हों", "किन्हों", "किन्हें", "इत्यादि", "इन्हों", "उन्हों", "बिलकुल", "निहायत", "इन्हीं", "उन्हीं", "जितना", "दूसरा", "कितना", "साबुत", 
"वग़ैरह", "कौनसा", "लिये", "दिया", "जिसे", "तिसे", "काफ़ी", "पहले", "बाला", "मानो", "अंदर", "भीतर", "पूरा", "सारा", "उनको", "वहीं", "जहाँ", 
"जीधर", "के", "एवं", "कुछ", "कुल", "रहा", "जिस", "जिन", "तिस", "तिन", "कौन", "किस", "संग", "यही", "बही", "उसी", "मगर", "कर", 
"मे", "एस", "उन", "सो", "अत", "पर  ", "इन ", "वह ", "यिह ", "वुह ", "जिन्हें", "जिन्हों", "तिन्हें", "तिन्हों", "किन्हों", "किन्हें", 
"इत्यादि", "द्वारा", "इन्हें", "इन्हों", "उन्हों", "बिलकुल", "निहायत", "ऱ्वासा", "इन्हीं", "उन्हीं", "उन्हें", "इसमें", "जितना", "दुसरा", "कितना", 
"दबारा", "साबुत", "वग़ैरह", "दूसरे", "कौनसा", "लेकिन", "होता", "करने", "किया", "लिये", "अपने", "नहीं", "दिया", "इसका", "करना", "वाले", 
"सकते", "इसके", "सबसे", "होने", "करते", "बहुत", "वर्ग", "करें", "होती", "अपनी", "उनके", "कहते", "होते", "करता", "उनकी", "इसकी", 
"सकता", "रखें", "अपना", "उसके", "जिसे", "तिसे", "किसे", "किसी", "काफ़ी", "पहले", "नीचे", "बाला", "यहाँ", "जैसा", "जैसे", "मानो", "अंदर", 
"भीतर", "पूरा", "सारा", "होना", "उनको", "वहाँ", "वहीं", "जहाँ", "जीधर", "उनका", "इनका", "के", "हैं", "गया", "बनी", "एवं", "हुआ", "साथ", 
"बाद", "लिए", "कुछ", "कहा", "यदि", "हुई", "इसे", "हुए", "अभी", "सभी", "कुल", "रहा", "रहे", "इसी", "उसे", "जिस", "जिन", "तिस", 
"तिन", "कौन", "किस", "कोई", "ऐसे", "तरह", "किर", "साभ", "संग", "यही", "बही", "उसी", "फिर", "मगर", "का", "एक", "यह", "से", 
"को", "इस", "कि", "जो", "कर", "मे", "ने", "तो", "ही", "या", "हो", "था", "तक", "आप", "ये", "थे", "दो", "वे", "थी", "जा", "ना", 
"उस", "एस", "पे", "उन", "सो", "भी", "और", "घर", "तब", "जब", "अत", "व", "न"
]

stopwords.extend(stopwords_list2)
stopwords.extend(stopwords_list3)
stopwords = set(stopwords)


# In[ ]:


import re
def stopword_removal(text):
  updated_text = []

  for i in text:
    i = re.sub('\d', "", i)
    if len(i) == 0:
      continue
    if i not in stopwords:
      if len(i)>4:
        updated_text.append(i)
  return updated_text

import string
p_translator = str.maketrans('', '', string.punctuation + "।“—०")
def tokeniser(text):
  text = text.translate(p_translator)
  return text.split()

def preprocessor(text):
  text = tokeniser(text)
  text = stopword_removal(text)
  return " ".join(text)


# In[ ]:


names =[]
import os
files = os.listdir("./ner_data/ner_data/")
for file in files:
  with open(f"./ner_data/ner_data/{file}") as f:
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
    if it in names:
      text[i]="<नाम>"
  return " ".join(text)


# In[ ]:


def remove_duplicates(data):
  data = dict(sorted(data.items(),key=lambda x:x[1],reverse = False))
  data_unique = {}
  last_text = ''
  for case, res in data.items():
    if res == last_text:
      continue 
    data_unique[case] = res 
    last_text = res 

  data_without_stop_words = {}
  for case, res in data_unique.items():
    data_without_stop_words[case] = preprocessor(res)
  
  data_without_stop_words = dict(sorted(data_without_stop_words.items(),key=lambda x:x[1],reverse = False))
  last_text = ''
  data_ret = {}
  for case, res in data_without_stop_words.items():
    if res == last_text:
      continue 
    data_ret[case] = data[case] 
    last_text = res
  return data_ret


# In[ ]:


def remove_invalids(data):
  def check_english(text):
    english_chars = 0
    total_chars = 0
    for c in text:
      total_chars += 1
      if 'a' <= c <= 'z' or 'A' <= c <= 'Z':
        english_chars += 1
    if 100 * (english_chars / total_chars) >= 50:
      return True
    else:
      return False

  BLANK_THRESH = 2**5
  LONG_THRESH = 2**20
  SHORT_THRESH = 2**10
  def check_valid(data):
    # Blank Documents
    if len(data) < BLANK_THRESH:
      return 'invalid', 'blank'

#     Too Long
    if len(data) > LONG_THRESH:
      return 'invalid', 'too long'
    
    # Language check
    if check_english(data):
      return 'invalid', 'english document'
    
    return 'valid', ''

  total_valid = 0
  total_invalid = 0
  valid_data = {}
  for court in data.keys():
    len1 = len(data[court])
    data[court] = remove_duplicates(data[court])
    len2 = len(data[court])
    valid_data[court] = {}
    valid_data[court]['valid'] = {}
    valid_data[court]['invalid'] = {}
    valid_data[court]['stats'] = defaultdict(int) 
    valid_data[court]['stats']['duplicates'] = len1-len2
    print('Removing Invalids')
    for case, res in data[court].items():
      if case in valid_data[court]['invalid'].keys():
        continue
      ret = check_valid(res)
      valid_data[court]['stats'][ret[0]] += 1
      if ret[0] != 'valid':
        total_invalid += 1
        valid_data[court]['invalid'][case] = res
        valid_data[court]['stats'][ret[1]] += 1
      else:
        total_valid += 1
        valid_data[court]['valid'][case] = res
    print('-'*150)
    print(f"{court}")
    print(json.dumps(valid_data[court]['stats'], indent=2))
  print('-'*120)
  print(f"total valid: {total_valid}, total invalid: {total_invalid}")
  return valid_data


# In[ ]:


import re
def check_with_improper_spacing(token, text):
  text = re.sub('\n', '', text)
  text = re.sub('\t', '', text)
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
                    'पर्याप्त आधार प्रतीत नहीं होता', 'खारिज किया जाता', 'अस्वीकार']


# In[ ]:


from collections import defaultdict 
processed_documents = []
unprocessed_documents = []

failed = ('#', '#')
thaty = 'तथ्य'
adesh = 'आदेश'
atah = 'अत:'
atah1 = 'अतः'
dhara = 'धारा'
thana = 'थाना'
viram = '|।'
words = [thaty, atah, atah1]

def check(text):
  for token in granted_tokens:
    if check_with_improper_spacing(token, text):
      return True
  for token in dismissed_tokens:
    if check_with_improper_spacing(token, text):
      return True
  return False

def divide(text):
  try:
    for word in words:
      if text.count(word) == 0:
        continue
      word_indices = [i for i in range(len(text)) if text.startswith(word, i)]
      for id in range(len(word_indices)):
        id = len(word_indices) - id
        if word == thaty:
          part2 = 'अतः मामले के समस्त तथ्य' +  word.join(text.split(word)[id:])
        else:
          part2 = 'अतः' +  word.join(text.split(word)[id:])
        part1_temp = word.join(text.split(word)[:id])
        last_purn_viram = len(part1_temp) - 1
        while part1_temp[last_purn_viram] not in viram:
          last_purn_viram -= 1
        part1 = part1_temp[:last_purn_viram+1]
        if check(part2):
          return part1, part2
    return failed
  except:
    return failed


def from_here(text, pos):
  while pos < len(text) and text[pos] not in viram:
    pos += 1
  body = text[pos+1:]
  s, i  = '', 0
  while i < len(body) and body[i] not in viram:
    s += body[i]
    i += 1
  i += 1
  if dhara in s:
    pos += i
  return text[:pos+1], text[pos+1:]

def split_header(text):
  d_indices = [i for i in range(len(text)) if text.startswith(dhara, i)]
  t_indices = [i for i in range(len(text)) if text.startswith(thana, i)]
  if len(t_indices) == 0:
    return '', text
  if len(d_indices) == 0:
    return from_here(text, t_indices[0])
  for id in t_indices:
    if id > d_indices[0]:
      return from_here(text, id)
  return '', text


# In[ ]:


def check_split(header, body, result):
  BLANK_THRESH = 32
  if len(body) < BLANK_THRESH:
    return False
  return True 


# In[ ]:


import re 
import textwrap
viram = '।|'
def check_break(txt):
  if txt[0] != '\n':
    return False
  if txt[1] == '\n':
    return True 
  if txt[1] in '123456789०१२३४५६७८९' and txt[2] in '.-':
    return True
  return False   

def divide_into_paragraphs(text):
  id = 0
  paras = []
  while id < len(text):
    s = ''
    while id < len(text) and not check_break(text[id:id + 3]):
      s += text[id]
      id += 1
    paras.append(s)
    id += 1
  cleaned_paras = []
  for para in paras:
    para = re.sub('\n', '', para)
    para = re.sub('\t', '', para)
    if len(para) == 0:
      continue 
    if len(cleaned_paras) > 0 and cleaned_paras[-1][-1] not in viram:
      cleaned_paras[-1] += para 
    else:
      cleaned_paras.append(para)
  return cleaned_paras


# In[ ]:


def check_anticipatory(text):
  token = 'अग्रिम जमानत प्रार्थनापत्र'
  if token in text:
    return True
  else:
    return False 


# In[ ]:


import random
import textwrap
def query(data):
  courts = list(data.keys())
  for court in courts:
    cases = list(data[court]['valid'].keys())
    if len(cases) <= 0:
      continue
    print(court)
    docs = random.sample(cases, 10)
    for case in docs:
      print('--------------------------'*10)
      print(case)
      print(data[court]['valid'][case])
    break

# ----------------------------------------------------------------------------
def document_split(district):  
  print('Entered: ', district)
  # ----------READING DATA------------------------------------------
  def only_bail_cases(data):
    return data
  print("OK")  
  def read_data():
    print("OK READ")  
    data = {}
    for court, path in district_files[district].items():
      with open(path, 'rb') as f:
        data[court] = only_bail_cases(pickle.load(f))
        for case, res in data[court].items():
          try:
            data[court][case] = res.decode('utf-8')
          except:
            data[court][case] = res
    return data
  
  data = read_data()
  
  # ------------REMOVING INVALID------------------------------------------
  data = remove_invalids(data)
  print('Removed Invalid')
    
# REMOVING Named entity
  for court in data.keys():
    for key in data[court].keys():
      if key == "invalid":
        continue
      if key == 'valid':
        for case in (data[court][key].keys()):
            data[court]['valid'][case] = remove_NER(data[court]['valid'][case])

# ------------STORING DATA---------------------------------------------------
  import os

  def get_file():
    return f"{HOME}/{district}/full_data_after_simple_NER_division.json"

  try:
    with open(get_file(), 'w') as f:
      json.dump(data, f)
  except Exception as e:
    print(f'exception: {district}')
    print("An exception occurred: ", e) 

  print(f"Done: {district}")
  print(f'-'*150)


# In[ ]:


from multiprocessing import Pool
with Pool(20) as p:
  for _ in tqdm(p.map(document_split, district_chunks[0]), total=len(district_chunks[0])):
    pass


# In[ ]:




