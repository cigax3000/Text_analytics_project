#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 17:44:33 2018

@author: MarioAntao
"""
import pickle
import pandas as pd

Digital_Music_sample_dict=pd.read_pickle('Digital_Music_sample_dict_lab.pickle')

baby_sample_dict=pd.read_pickle('baby_sample_dict_lab.pickle')

cd_and_vinyl_sample_dict=pd.read_pickle('cd_and_vinyl_sample_dict_lab.pickle')

Toys_games_dict=pd.read_pickle('Toys_games_dict_lab.pickle')



def unix(dictionairy):
    from datetime  import datetime
    import datetime
    for item in dictionairy:
            item['unixReviewTime']=datetime.datetime.fromtimestamp(
            item ['unixReviewTime']
            ).strftime('%Y-%m-%d %H:%M:%S')        

"""
Split test and train data
"""
from sklearn.cross_validation import train_test_split

#Split Digital Music
review_dig_train,review_dig_test=train_test_split(Digital_Music_sample_dict,test_size=0.3)
#Split Baby
review_baby_train,review_baby_test=train_test_split(baby_sample_dict,test_size=0.3)
#Split Cd and Vinyl
review_cd_train,review_cd_test=train_test_split(cd_and_vinyl_sample_dict,test_size=0.3)
#Split Toys and Games
review_toys_train,review_toys_test=train_test_split(Toys_games_dict,test_size=0.3)


#Merge Train data
merged_train_dic = review_dig_train + review_baby_train + review_cd_train + review_toys_train

#Merge Test Data
merged_test_dic = review_dig_test + review_baby_test + review_cd_test + review_toys_test

#Convert unix to date
unix(merged_test_dic)
unix(merged_train_dic)




"""Export Data with pickle"""
with open('merged_train_dic.pickle', 'wb') as handle:
    pickle.dump(merged_train_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
"""Export Data with pickle"""
with open('merged_test_dic.pickle', 'wb') as handle:
    pickle.dump(merged_test_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
   


merged_train_dic=pd.read_pickle('merged_train_dic.pickle')
merged_test_dic=pd.read_pickle('//Users//MarioAntao//Documents//Try_1//merged_test_dic.pickle')
import random

random.shuffle(merged_train_dic)
random.shuffle(merged_test_dic)


