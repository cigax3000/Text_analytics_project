#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 18:31:29 2018

@author: MarioAntao
"""

import scipy.sparse as sp
from numpy.linalg import norm
feature_names = bow_vectorizer.get_feature_names()

# compute term frequency
tf = bow_features
tf = np.array(tf, dtype='float64')

# show term frequencies
display_features(tf, feature_names)

# build the document frequency matrix
df = np.diff(sp.csc_matrix(bow_features, copy=True).indptr)
df = 1 + df # to smoothen idf later

# show document frequencies
display_features([df], feature_names)

# compute inverse document frequencies
total_docs = 1 + len(CORPUS)
idf = 1.0 + np.log(float(total_docs) / df)

# show inverse document frequencies
display_features([np.round(idf, 2)], feature_names)

# compute idf diagonal matrix  
total_features = bow_features.shape[1]
idf_diag = sp.spdiags(idf, diags=0, m=total_features, n=total_features)
idf = idf_diag.todense()

# print the idf diagonal matrix
print np.round(idf, 2)

# compute tfidf feature matrix
tfidf = tf * idf

# show tfidf feature matrix
display_features(np.round(tfidf, 2), feature_names)

# compute L2 norms 
norms = norm(tfidf, axis=1)

# print norms for each document
print np.round(norms, 2)

# compute normalized tfidf
norm_tfidf = tfidf / norms[:, None]

# show final tfidf feature matrix
display_features(np.round(norm_tfidf, 2), feature_names)
 

# compute new doc term freqs from bow freqs
nd_tf = new_doc_features
nd_tf = np.array(nd_tf, dtype='float64')

# compute tfidf using idf matrix from train corpus
nd_tfidf = nd_tf*idf
nd_norms = norm(nd_tfidf, axis=1)
norm_nd_tfidf = nd_tfidf / nd_norms[:, None]

# show new_doc tfidf feature vector
display_features(np.round(norm_nd_tfidf, 2), feature_names)


from feature_extractors import tfidf_extractor
    
tfidf_vectorizer, tdidf_features = tfidf_extractor(CORPUS)
display_features(np.round(tdidf_features.todense(), 2), feature_names)

nd_tfidf = tfidf_vectorizer.transform(new_doc)
display_features(np.round(nd_tfidf.todense(), 2), feature_names)    


import gensim
import nltk

TOKENIZED_CORPUS = [nltk.word_tokenize(sentence) 
                    for sentence in CORPUS]
tokenized_new_doc = [nltk.word_tokenize(sentence) 
                    for sentence in new_doc]                        

model = gensim.models.Word2Vec(TOKENIZED_CORPUS, 
                               size=10,
                               window=10,
                               min_count=2,
                               sample=1e-3)


from feature_extractors import averaged_word_vectorizer


avg_word_vec_features = averaged_word_vectorizer(corpus=TOKENIZED_CORPUS,
                                                 model=model,
                                                 num_features=10)
print np.round(avg_word_vec_features, 3)

nd_avg_word_vec_features = averaged_word_vectorizer(corpus=tokenized_new_doc,
                                                    model=model,
                                                    num_features=10)
print np.round(nd_avg_word_vec_features, 3)

              
from feature_extractors import tfidf_weighted_averaged_word_vectorizer

corpus_tfidf = tdidf_features
vocab = tfidf_vectorizer.vocabulary_
wt_tfidf_word_vec_features = tfidf_weighted_averaged_word_vectorizer(corpus=TOKENIZED_CORPUS,
                                                                     tfidf_vectors=corpus_tfidf,
                                                                     tfidf_vocabulary=vocab,
                                                                     model=model, 
                                                                     num_features=10)
print np.round(wt_tfidf_word_vec_features, 3)

nd_wt_tfidf_word_vec_features = tfidf_weighted_averaged_word_vectorizer(corpus=tokenized_new_doc,
                                                                     tfidf_vectors=nd_tfidf,
                                                                     tfidf_vocabulary=vocab,
                                                                     model=model, 
                                                                     num_features=10)
print np.round(nd_wt_tfidf_word_vec_features, 3)   
