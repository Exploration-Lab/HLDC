#!/usr/bin/env python
# coding: utf-8

# In[14]:


import json  
import zipfile  
PATH_TO_ZIP = 'full_bail_data_with_info_extract_varanasi_released_data.zip'
d = None  
data = None  

with zipfile.ZipFile(PATH_TO_ZIP, 'r') as z:
    
    for filename in z.namelist():  
        print('------ Main File ------')
        print(filename) 
        print('\n')
        
        
        with z.open(filename) as f:  
            data = f.read()  
            d = json.loads(data)  
            print('------ Each District Courts ------')
        
            for courts in d:
                print(courts)
            my_court = 'varanasi_district_court_complex'
            print('\n')
        
            print('------ Subheadings for each data point ------')
        
            for headings in d[my_court]:
                print(headings)
                
            print('\n')
            
            for case_number in d[my_court]['processed']:
                print('-------- Case Number ----')
                print(case_number)
                print('\n')
        
                
                print('------- Division of each case into respective fields ----')
                
                print(d[my_court]['processed'][case_number].keys())
                print('\n')
        
                
                break


# In[ ]:




