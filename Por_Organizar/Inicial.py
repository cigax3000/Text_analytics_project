#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 15:33:03 2018

@author: MarioAntao
"""

def column(matrix, i):
    return [row[i] for row in matrix]
 import pandas as pd

baby_sample=pd.read_pickle('baby_sample.pickle')

(create_dict_from_list(review_list) for review_list in review_list_list)


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

print(create_dict_from_list(baby_sample[0]))

baby_sample_dict=create_dict_from_list(baby_sample)
type(baby_sample_dict[0])

baby_sample[:][7]

baby_sample[0]


my_dicts = [create_dict_from_list(baby) for baby in baby_sample]


baby_sample_dict=lst

baby_sample_dict1=create_dict_from_list(baby_sample)

type(baby_sample_dict[0])