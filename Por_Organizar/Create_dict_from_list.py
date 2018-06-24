#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 16:20:26 2018

@author: MarioAntao
"""


import pickle
import pandas as pd

#Digital_Music_Sample
    
    

Digital_Music_sample=pd.read_pickle('Digital_Music_Sample.pickle')


def create_dict_from_list (review_list):
    review_dict = {
            "reviewer_id": review_list[0],
            "product_id": review_list[1],
              "helpful":review_list[3],
            "reviewText":review_list[4],
            "overall":review_list[5],
            "summary":review_list[6],
            "unixReviewTime":review_list[7]
          
            } 
    
    return review_dict

Digital_Music_dict = [create_dict_from_list(item) for item in Digital_Music_sample]


"""Export Data with pickle"""
with open('Digital_Music_Sample_dict.pickle', 'wb') as handle:
    pickle.dump(Digital_Music_Sample_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
#Baby_Sample
Digital_Music_sample=pd.read_pickle('baby_sample.pickle')


def create_dict_from_list (review_list):
    review_dict = {
            "reviewer_id": review_list[0],
            "product_id": review_list[1],
              "helpful":review_list[3],
            "reviewText":review_list[4],
            "overall":review_list[5],
            "summary":review_list[6],
            "unixReviewTime":review_list[7]
          
            } 
    
    return review_dict

baby_sample_dict = [create_dict_from_list(item) for item in baby_sample]


"""Export Data with pickle"""
with open('baby_sample_dict.pickle', 'wb') as handle:
    pickle.dump(baby_sample_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
#Toys_games_Sample
Toys_games_sample=pd.read_pickle('Toys_games_sample.pickle')


def create_dict_from_list (review_list):
    review_dict = {
            "reviewer_id": review_list[0],
            "product_id": review_list[1],
              "helpful":review_list[3],
            "reviewText":review_list[4],
            "overall":review_list[5],
            "summary":review_list[6],
            "unixReviewTime":review_list[7]
          
            } 
    
    return review_dict

Toys_games_dict = [create_dict_from_list(item) for item in Toys_games_sample]


"""Export Data with pickle"""
with open('Toys_games_dict.pickle', 'wb') as handle:
    pickle.dump(Toys_games_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
#Cd_and_Vinyl_Sample
cd_and_vinyl_sample=pd.read_pickle('cd_and_vinyl_sample.pickle')


def create_dict_from_list (review_list):
    review_dict = {
            "reviewer_id": review_list[0],
            "product_id": review_list[1],
              "helpful":review_list[3],
            "reviewText":review_list[4],
            "overall":review_list[5],
            "summary":review_list[6],
            "unixReviewTime":review_list[7]
          
            } 
    
    return review_dict

cd_and_vinyl_sample_dict = [create_dict_from_list(item) for item in cd_and_vinyl_sample]


"""Export Data with pickle"""
with open('cd_and_vinyl_sample_dict.pickle', 'wb') as handle:
    pickle.dump(cd_and_vinyl_sample_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)





