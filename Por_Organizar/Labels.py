#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 16:49:08 2018

@author: MarioAntao
"""

import pickle
import pandas as pd
#Digital_Music

Digital_Music_sample_dict=pd.read_pickle('Digital_Music_Sample_dict.pickle')

for item in Digital_Music_sample_dict:
    item.update( {"category":"digital_music"})

"""Export Data with pickle"""
with open('Digital_Music_sample_dict_lab.pickle', 'wb') as handle:
    pickle.dump(Digital_Music_sample_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
#Baby_Sample


baby_sample_dict=pd.read_pickle('baby_sample_dict.pickle')

for item in baby_sample_dict:
    item.update( {"category":"baby"})

"""Export Data with pickle"""
with open('baby_sample_dict_lab.pickle', 'wb') as handle:
    pickle.dump(baby_sample_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#CD and Vinyl    

cd_and_vinyl_sample_dict=pd.read_pickle('cd_and_vinyl_sample_dict.pickle')

for item in cd_and_vinyl_sample_dict:
    item.update( {"category":"cd_and_vinyl"})

"""Export Data with pickle"""
with open('cd_and_vinyl_sample_dict_lab.pickle', 'wb') as handle:
    pickle.dump(cd_and_vinyl_sample_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#Toys and Games

Toys_games_dict=pd.read_pickle('Toys_games_dict.pickle')

for item in Toys_games_dict:
    item.update( {"category":"toys_games"})

"""Export Data with pickle"""
with open('Toys_games_dict_lab.pickle', 'wb') as handle:
    pickle.dump(Toys_games_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    