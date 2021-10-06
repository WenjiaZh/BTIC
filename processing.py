#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 14:45:20 2021

@author: wjz
"""
import os
import pandas as pd
import numpy as np
import spacy
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split






df = pd.read_csv('source path + recovery-news-data.csv')
imagepath = 'load images from'
datapath = 'load data from'
savepath = 'save to'



### pick up news that has available image ###
listName = []
for fileName in os.listdir(imagepath):
    if os.path.splitext(fileName)[1] == '.jpg':
        fileName = os.path.splitext(fileName)[0]
        listName.append(fileName)

nid = pd.DataFrame(listName,dtype=np.int, columns=['news_id'])
ndf = pd.merge(df,nid,how='inner',on='news_id')
df1 = ndf[ndf['reliability']==1].reset_index(drop=True)
df0 = ndf[ndf['reliability']==0].reset_index(drop=True)

df_new =  pd.DataFrame()
df_new['Id'] = ndf['news_id']  
df_new['label'] = ndf['reliability'] 
df_new['title'] = ndf['title'] 
df_new['body_text'] = ndf['body_text'] 
df_new['titletext'] = df_new['title'] + ". " + df_new['body_text']

df_new.to_csv(datapath + '/text.csv', index=False)


###find reference sample ID ###
spload = spacy.load('en_core_web_trf')
k = 5
k1 = df1['news_id'][k]
k0 = df0['news_id'][k]
s = df_new.loc[df_new.isin([min(k1,k0)]).any(axis=1)].index[0]

reff = pd.DataFrame()
for i in range(s,len(df_new)):
    I = df_new['Id'][i]
    ref1 = df1[(df1.news_id<I)]
    ref0 = df0[(df0.news_id<I)]
    title_I = df_new['title'][i]
    doc = spload(title_I)
    ref1_sim = []
    for j in range(len(ref1)):
        t_j = spload(ref1['title'][j])
        sim = doc.similarity(t_j)
        ref1_sim.append(sim)
        
    ref0_sim = []
    for l in range(len(ref0)):
        t_l = spload(ref0['title'][l])
        sim = doc.similarity(t_l)
        ref0_sim.append(sim)    
    
    ref1['sim'] = ref1_sim
    ref0['sim'] = ref0_sim
    
    ref1_id = ref1.sort_values(by=['sim'],ascending = False).head(k)['news_id']
    ref1_doby_text = ref1.sort_values(by=['sim'],ascending = False).head(k)['body_text']
    ref1 = pd.concat([ref1_id,ref1_doby_text],axis=0)
    new_col_1 = ['i11','i12','i13','i14','i15','t11','t12','t13','t14','t15']
    ref1_list = ref1.to_frame().T
    ref1_list.columns = new_col_1 
    
    ref0_id = ref0.sort_values(by=['sim'],ascending = False).head(k)['news_id']
    ref0_doby_text = ref0.sort_values(by=['sim'],ascending = False).head(k)['body_text']  
    ref0 = pd.concat([ref0_id,ref0_doby_text],axis=0)
    new_col_0 = ['i01','i02','i03','i04','i05','t01','t02','t03','t04','t05']
    ref0_list = ref0.to_frame().T
    ref0_list.columns = new_col_0
    
    ref = pd.concat([ref1_list,ref0_list],axis=1).reset_index(drop=True)
    ref['Id'] = I
    reff = reff.append(ref)

df_data =  pd.merge(df_new,reff,how='inner',on='Id')

df_data.to_csv(datapath + '/new_data.csv', index=False)


###split data time_order and shuffled

###time_order
df_data = pd.read_csv(datapath+ '/new_data.csv')

def split(file, split_ratio):
        total = len(file)
        test_n = int(split_ratio*total)
        train_file = file[:test_n]
        test_file = file[test_n:]
        return train_file, test_file
    
ratio = 0.80


df_real = df_data[df_data['label'] == 1]
df_fake = df_data[df_data['label'] == 0]
df_real_full_train, df_real_test = split(df_real, split_ratio = ratio)
df_fake_full_train, df_fake_test = split(df_fake, split_ratio = ratio)
df_real_train, df_real_valid = split(df_real_full_train, split_ratio = ratio)
df_fake_train, df_fake_valid = split(df_fake_full_train, split_ratio = ratio)

df_train = pd.concat([df_real_train, df_fake_train], ignore_index=True, sort=False)
df_valid = pd.concat([df_real_valid, df_fake_valid], ignore_index=True, sort=False)
df_test = pd.concat([df_real_test, df_fake_test], ignore_index=True, sort=False)

# train = df_train.sort_values(by=['Id'], ascending=True).reset_index(drop=True)
# valid = df_valid.sort_values(by=['Id'], ascending=True).reset_index(drop=True)
train = shuffle(df_train,random_state=3)
valid = shuffle(df_valid,random_state=3)
test = shuffle(df_test,random_state=3)

train.to_csv(datapath + '/timeorder/train.csv', index=False)
valid.to_csv(datapath + '/timeorder/valid.csv', index=False)
test.to_csv(datapath + '/timeorder/test.csv', index=False)

###shuffled

df_data = pd.read_csv(datapath+ '/new_data.csv')

train_test_ratio = 0.80
train_valid_ratio = 0.80

df_real = df_data[df_data['label'] == 1]
df_fake = df_data[df_data['label'] == 0]
df_real_full_train, df_real_test = train_test_split(df_real, train_size = train_test_ratio, random_state = 1)
df_fake_full_train, df_fake_test = train_test_split(df_fake, train_size = train_test_ratio, random_state = 1)
df_real_train, df_real_valid = train_test_split(df_real_full_train, train_size = train_valid_ratio, random_state = 1)
df_fake_train, df_fake_valid = train_test_split(df_fake_full_train, train_size = train_valid_ratio, random_state = 1)

df_train = pd.concat([df_real_train, df_fake_train], ignore_index=True, sort=False)
df_valid = pd.concat([df_real_valid, df_fake_valid], ignore_index=True, sort=False)
df_test = pd.concat([df_real_test, df_fake_test], ignore_index=True, sort=False)

train = shuffle(df_train,random_state=3)
valid = shuffle(df_valid,random_state=3)
test = shuffle(df_test,random_state=3)

train.to_csv(datapath + '/shuffled/train.csv', index=False)
valid.to_csv(datapath + '/shuffled/valid.csv', index=False)
test.to_csv(datapath + '/shuffled/test.csv', index=False)















