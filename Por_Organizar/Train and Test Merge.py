#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 18:54:27 2018

@author: MarioAntao
"""



from sklearn.cross_validation import train_test_split

merged_train,merged_test=train_test_split(merged_dic,test_size=0.3)

"""Export Data with pickle"""
with open('merged_train.pickle', 'wb') as handle:
    pickle.dump(merged_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""Export Data with pickle"""
with open('merged_test.pickle', 'wb') as handle:
    pickle.dump(merged_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    