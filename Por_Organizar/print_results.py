#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 19:59:47 2018

@author: MarioAntao
"""

print('BAG OF WORDS MODELS')
print('SVM with tf-idf F-statistic:')
print(get_metrics(test_label,SVM_bow_f))
print(' ')
print('SVM with tf-idf Chi2:')
print(get_metrics(test_label,SVM_bow_chi2))
print(' ')
print('SVM with tf-idf Lsa with 5 concepts:')
print(get_metrics(test_label,SVM_bow_lsa_5))
print(' ')
print(' ')
print('BIGRAMS MODELS')
print('SVM with tf-idf F-statistic:')
print(get_metrics(test_label,SVM_big_f))
print(' ')
print('SVM with tf-idf Chi2:')
print(get_metrics(test_label,SVM_big_chi2))
print(' ')
print('SVM with tf-idf Lsa with 5 concepts:')
print(get_metrics(test_label,SVM_big_lsa_5))
print(' ')
print(' ')
print('BAG OF WORDS ONLY WITH NOUNS MODELS:')
print('SVM with tf-idf F-statistic:')
print(get_metrics(test_label,SVM_bown_f))
print(' ')
print('SVM with tf-idf Chi2:')
print(get_metrics(test_label,SVM_bown_chi2))
print(' ')
print('SVM with tf-idf Lsa with 5 concepts:')
print(get_metrics(test_label,SVM_bown_lsa_5))


