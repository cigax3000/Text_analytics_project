#from  evaluation import metrics
from  src.features.BOW_Bigrams.BOW_Tokenization import word_tokenize, ngrams_tokenize, vocabulary_size
from  src.features.process_text.feature_weighting import compute_tfidf, compute_tfidf_stopwords, compute_tfidfle_stopwords
from  src.features.LSA_1 import lsa_try,elbow_lsa,fit_lsa
from  src.features.Metrics import get_metrics,train_predict_evaluate_model
from  src.models.supervised_classification import SVC_model,look_at_predictions, prediction
#from  src.utils.look_tfidfs import look_at_features


['problem', 'help', 'function', 'one']
def main():
    """
    The objective of this class is to investigate different techniques to:
        1. Use sklearn to explore data
        2. Lemmatization with sklearn
        3. TF-IDF
        4. Supervised learning
        5. Evaluation metrics
    """


    """
    SECTION  2:
        Look at an example were we do BOW and n-grams(=2)
            2.1 Use the function from features.process_text.lemmatization to compute lemmas from the dataset
                a) Look at data (similar to point 2)
            2.2. Compare the vocabulary size with and without using lemmas.
                a) Is it the same as before we applied the lemmas? Yes/No what happen?

    """
"""
def display_features(features, feature_names):
   df = pd.DataFrame(data=features,
                      columns=feature_names)
    return df
"""    
    
    """
    BOW
    """
       # Tokenization_train (BOW)
        bow_tr_features, bow_tr_vectorizer = word_tokenize(train_corpus)  # data.data = data_lst.tolist()
     
        
        #bow_transformer, bow_features = compute_tfidf(bow_index)
      
      #Tokenization_test (BOW)
        #bow_te_features, bow_te_vectorizer = word_tokenize(te_corpus_sample)  # data.data = data_lst.tolist()
        bow_te_features = bow_tr_vectorizer.transform(test_corpus) 
    
              
        #Vocabulary size
     # print('Vocabulary size (bow): '+str(vocabulary_size(bow_vectorizer)))
     
        #Look at some of the vocabulary
      #print(bow_vectorizer.get_feature_names()[90:1000])
     
 #Get features name
   
   bow_tr_features_name = bow_tr_vectorizer.get_feature_names()
   
   #bow_te_features_name = bow_te_vectorizer.get_feature_names()
   #display_features(bow_features,bow_features_names)
   
   
"""
BIAGRAMS
"""

     
        #Tokenization train (bigrams)
      bigram_tr_features, bigram_tr_vectorizer = ngrams_tokenize(train_corpus, 2, 2)
      #Save Bigrams      
      big_tr_features_name = bigram_tr_vectorizer.get_feature_names()
      
      
      #Tokenization test (bigrams)
      
      bigram_te_features = bigram_tr_vectorizer.transform(test_corpus) 

      #bigram_te_features, bigram_te_vectorizer = ngrams_tokenize(te_corpus_sample, 1, 2)

     """ 
        #Vocabulary size
      print('Vocabulary size (bigrams): '+str(vocabulary_size(bigram_vectorizer)))
      perc = round(float(vocabulary_size(bigram_vectorizer)*100)/vocabulary_size(bow_vectorizer), 2)
      print('Bigrams vocabulary size is '+str(perc)+'% larger than bow')
     
      #  Look at some of the vocabulary
      print(bigram_vectorizer.get_feature_names()[200000:200010])
      
      """
      
      
"""
BOW NOUNS
"""

        # BOW only Nouns
        


def extract_nouns(list_of_strings):
    l=[]
    nouns=[]
    for word in list_of_strings:
        word_tagged = nltk.pos_tag(word.split())
        l.append([w for (w, t) in word_tagged if t.startswith("N")])
    
    nouns=[' '.join(x) for x in l]
    return nouns

import nltk

bow_n_tr = extract_nouns(train_corpus)
bow_n_te = extract_nouns(test_corpus)

"""Export Data with pickle"""
with open('bow_n_tr.pickle', 'wb') as handle:
    pickle.dump(bow_n_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""Export Data with pickle"""
with open('bow_n_te.pickle', 'wb') as handle:
    pickle.dump(bow_n_te, handle, protocol=pickle.HIGHEST_PROTOCOL)

      #Train
      bown_tr_features, bown_tr_vectorizer = word_tokenize(bow_n_tr)  # data.data = data_lst.tolist()
      bown_tr_features_name = bown_tr_vectorizer.get_feature_names()
      
      #Test
      bown_te_features = bown_tr_vectorizer.transform(bow_n_te)
      
      bown_te_features_name = bown_te_vectorizer.get_feature_names()



    """
    SECTION  3:
        TF-IDF
            3.1 Look at the functions: tfidf, tfidf_lemma_stopwords ( src.features.process_text.feature_weighting)
            3.2 Run the bellow code. Notice that if we use stopwords or lemmatization TFIDF weights change. Why is that?
    WARNING: THIS SECTION CAN TAKE A LONG TIME TO COMPUTE!
    """
    
    
    
#Tdif BOW
    
#Train
    #bow_transformer, bow_features = compute_tfidf(bow_index)
    
      bow_tr_tfidf, bow_tr_idfs = compute_tfidf(bow_tr_features)
     
     # bow_tfidfle_sw, bow_idfsle_sw = compute_tfidfle_stopwords(bow_tr_features, stopwords_lang='english')
      
      #bow_tfidf_sw, bow_idfs_sw = compute_tfidf_stopwords(bow_tr_features,stopwords_lang='english)

#Test
      #bow_transformer, bow_features = compute_tfidf(bow_te_features)
    
      #bow_te_tfidf, bow_te_idfs = compute_tfidf(bow_te_features)
      
      bow_te_idfs = bow_tr_tfidf.transform(bow_te_features)

 
#Tdif Bigrams

#Train
      
      big_tr_tfidf, big_tr_idfs = compute_tfidf(bigram_tr_features)
     
      #big_tr_tfidfle_sw, big_tr_idfsle_sw = compute_tfidfle_stopwords(bigram_tr_features, stopwords_lang='english')
#Test
      
      big_te_idfs = big_tr_tfidf.transform(bigram_te_features)
      
      #big_te_tfidf, big_te_idfs = compute_tfidf(bigram_te_features)
     
      #big_te_idfsle_sw =  big_tr_tfidfle_sw.transform(bigram_te_features)
      


#Tdif BOW Nouns

#Train
      bown_tr_tfidf, bown_tr_idfs = compute_tfidf(bown_tr_features)
      
     #print(big_a_idfs)
     
      #bown_tr_tfidfle_sw, bown_tr_idfsle_sw = compute_tfidfle_stopwords(bown_tr_features, stopwords_lang='english')
#Test

      bown_te_idfs = bown_tr_tfidf.transform(bown_te_features)
      
      
      #bown_te_idfsle_sw = bown_tr_tfidfle_sw.transform(bown_te_features)
     
    #bown_te_tfidf, bown_te_idfs = compute_tfidf(bown_te_features)
    # bown_te_tfidfle_sw, bown_te_idfsle_sw = compute_tfidfle_stopwords(bown_te_features, stopwords_lang='english')





#import numpy as np
#features=np.round(bown_tfidf,2)
#display_features(features, bown_features_names)     


"""

Perform LSA

"""


#BOW Td_Idf
#Train
lsa_model_bow_tr, lsa_features_bow_tr = lsa_try(bow_tr_idfs,5,bow_tr_features_name)

#Graphic
lsa_sv_bow=lsa_model_bow_tr.singular_values_
elbow_lsa(lsa_sv_bow,30)


"""Export Data with pickle"""
with open('lsa_model_bow_tr', 'wb') as handle:
    pickle.dump(lsa_model_bow_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""Export Data with pickle"""
with open('lsa_features_bow_tr', 'wb') as handle:
    pickle.dump(lsa_features_bow_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#Cut-off   
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif

#F_classif
selector_bow_f = SelectKBest(f_classif, k=5)

bow_tfidf_f_tr = selector_bow_f.fit_transform(bow_tr_idfs,train_label)
bow_tfidf_f_te = selector_bow_f.transform(bow_te_idfs)

"""Export Data with pickle"""
with open('bow_tfidf_f_tr', 'wb') as handle:
    pickle.dump(bow_tfidf_f_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""Export Data with pickle"""
with open('bow_tfidf_f_te', 'wb') as handle:
    pickle.dump(bow_tfidf_f_te, handle, protocol=pickle.HIGHEST_PROTOCOL)
  
    
#Chi2
selector_bow_chi2 = SelectKBest(chi2, k=5)

bow_tfidf_chi2_tr = selector_bow_chi2.fit_transform(bow_tr_idfs,train_label)
bow_tfidf_chi2_te = selector_bow_chi2.transform(bow_te_idfs)

"""Export Data with pickle"""
with open('big_tfidf_chi2_tr', 'wb') as handle:
    pickle.dump(bow_tfidf_chi2_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""Export Data with pickle"""
with open('big_tfidf_chi2_te', 'wb') as handle:
    pickle.dump(bow_tfidf_chi2_te, handle, protocol=pickle.HIGHEST_PROTOCOL)


#Caso queiramos usar LSA
#Teste

lsa_features_bow_te = lsa_model_bow_tr.transform(bow_te_idfs)









#Bigrams_Td_Idf

lsa_model_big_tr, lsa_features_big_tr = lsa_try(big_tr_idfs,5,big_tr_features_name)

#Gráfico
lsa_sv_big=lsa_model_big_tr.singular_values_
elbow_lsa(lsa_sv_big,30)


"""Export Data with pickle"""
with open('lsa_model_big_tr_5', 'wb') as handle:
    pickle.dump(lsa_model_big_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""Export Data with pickle"""
with open('lsa_features_big_tr_5', 'wb') as handle:
    pickle.dump(lsa_features_big_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)


#Cut-off
    
#F_classif
selector_big_f = SelectKBest(f_classif, k=5)

big_tfidf_f_tr = selector_big_f.fit_transform(big_tr_idfs,train_label)
big_tfidf_f_te = selector_big_f.transform(big_te_idfs)

"""Export Data with pickle"""
with open('big_tfidf_f_tr', 'wb') as handle:
    pickle.dump(big_tfidf_f_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""Export Data with pickle"""
with open('big_tfidf_f_te', 'wb') as handle:
    pickle.dump(big_tfidf_f_te, handle, protocol=pickle.HIGHEST_PROTOCOL)
  
    
#Chi2
selector_big_chi2 = SelectKBest(chi2, k=5)

big_tfidf_chi2_tr = selector_big_chi2.fit_transform(big_tr_idfs,train_label)
big_tfidf_chi2_te = selector_big_chi2.transform(big_te_idfs)

"""Export Data with pickle"""
with open('big_tfidf_chi2_tr', 'wb') as handle:
    pickle.dump(big_tfidf_chi2_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""Export Data with pickle"""
with open('big_tfidf_chi2_te', 'wb') as handle:
    pickle.dump(big_tfidf_chi2_te, handle, protocol=pickle.HIGHEST_PROTOCOL)


#Caso queiramos usar LSA
#Teste

lsa_features_big_te = lsa_model_big_tr.transform(big_te_idfs)



#Bow_n_Td_Idf


lsa_model_bown_tr, lsa_features_bown_tr = lsa_try(bown_tr_idfs,5,bown_tr_features_name)

#Gráfico
lsa_sv_bown=lsa_model_bown_tr.singular_values_
elbow_lsa(lsa_sv_bown,30)


"""Export Data with pickle"""
with open('lsa_model_bown_tr', 'wb') as handle:
    pickle.dump(lsa_model_bown_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""Export Data with pickle"""
with open('lsa_features_bown_tr', 'wb') as handle:
    pickle.dump(lsa_features_bown_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)


#Cut-off
    
#F_classif
selector_bown_f = SelectKBest(f_classif, k=5)

bown_tfidf_f_tr = selector_bown_f.fit_transform(bown_tr_idfs,train_label)
bown_tfidf_f_te = selector_bown_f.transform(bown_te_idfs)

"""Export Data with pickle"""
with open('bown_tfidf_f_tr', 'wb') as handle:
    pickle.dump(bown_tfidf_f_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""Export Data with pickle"""
with open('bown_tfidf_f_te', 'wb') as handle:
    pickle.dump(bown_tfidf_f_te, handle, protocol=pickle.HIGHEST_PROTOCOL)
  
    
#Chi2
selector_bown_chi2 = SelectKBest(chi2, k=5)

bown_tfidf_chi2_tr = selector_bown_chi2.fit_transform(bown_tr_idfs,train_label)
bown_tfidf_chi2_te = selector_bown_chi2.transform(bown_te_idfs)

"""Export Data with pickle"""
with open('bown_tfidf_chi2_tr', 'wb') as handle:
    pickle.dump(bown_tfidf_chi2_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""Export Data with pickle"""
with open('bown_tfidf_chi2_te', 'wb') as handle:
    pickle.dump(bown_tfidf_chi2_te, handle, protocol=pickle.HIGHEST_PROTOCOL)


#Caso queiramos usar LSA
#Teste
lsa_features_bown_te = lsa_model_bown_tr.transform(bown_te_idfs)




    """
    SECTION  4:
        Supervised learning
            4.1 Compute the Multinominal Naive Bayes for BOW, Bigrams and TFIDF (with lemmas and stopwords)
            4.2 Implement SVM (TIP: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
    """


      
      svc_model=SVC_model(X_train_clean, tr_label_sample)
     
      predicted = prediction(svc_model, a, te_label_sample)
     
      look_at_predictions(predicted, tr_corpus_sample, te_corpus_sample)
"""
SVM
"""

    """
    BOW
    """

#SVM Bow tf_idf
     
    #F-classif
SVM_bow_f= train_predict_evaluate_model(bow_tfidf_f_tr,train_label,
                             bow_tfidf_f_te,test_label)

"""Export Data with pickle"""
with open('SVM_bow_f', 'wb') as handle:
    pickle.dump(SVM_bow_f, handle, protocol=pickle.HIGHEST_PROTOCOL)
 

SVM_bow_f=pd.read_pickle('SVM_bow_f')    
    
    
 # Compute confusion matrix
cnf_matrix = confusion_matrix(test_label, SVM_bow_f)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title=' SVM Bow F-class Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title=' SVM Bow F-class Normalized confusion matrix')

plt.show()   
    
    
    
    
    
     #Chi2   
    
    SVM_bow_chi2= train_predict_evaluate_model(bow_tfidf_chi2_tr,train_label,
                             bow_tfidf_chi2_te,test_label)
        
"""Export Data with pickle"""
with open('SVM_bow_chi2', 'wb') as handle:
    pickle.dump(SVM_bow_chi2, handle, protocol=pickle.HIGHEST_PROTOCOL)

SVM_bow_chi2=pd.read_pickle('SVM_bow_chi2')

# Compute confusion matrix
cnf_matrix = confusion_matrix(test_label, SVM_bow_chi2)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title=' SVM Bow Chi 2Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title=' SVM  Chi2 Bow Normalized confusion matrix')

plt.show()






#Lsa 5
        SVM_bow_lsa_5= train_predict_evaluate_model(lsa_features_bow_tr,train_label,
                             lsa_features_bow_te,test_label)
"""Export Data with pickle"""
with open('SVM_bow_lsa_5', 'wb') as handle:
    pickle.dump(SVM_bow_lsa_5, handle, protocol=pickle.HIGHEST_PROTOCOL)    


SVM_bow_lsa_5=pd.read_pickle('SVM_bow_lsa_5')

# Compute confusion matrix
cnf_matrix = confusion_matrix(test_label, SVM_bow_lsa_5)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title=' SVM Lsa Bow Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title=' SVM Lsa Bow Normalized confusion matrix')

plt.show()




    """
    Big
    """
    
    
    #Tf_idf
    
    #F-classif
    
    SVM_big_f= train_predict_evaluate_model(big_tfidf_f_tr,train_label,
                             big_tfidf_f_te,test_label)
"""Export Data with pickle"""
with open('SVM_big_f', 'wb') as handle:
    pickle.dump(SVM_big_f, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
 # Compute confusion matrix
cnf_matrix = confusion_matrix(test_label, SVM_big_f)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title=' SVM Bigrams F-class Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title=' SVM Bigrams F-class Normalized confusion matrix')

plt.show()  

    
    #Chi2

    SVM_big_chi2=train_predict_evaluate_model(big_tfidf_chi2_tr,train_label,
                             big_tfidf_chi2_te,test_label)

"""Export Data with pickle"""
with open('SVM_big_chi2', 'wb') as handle:
    pickle.dump(SVM_big_chi2, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Compute confusion matrix
cnf_matrix = confusion_matrix(test_label, SVM_big_chi2)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title=' SVM Bigrams Chi 2Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title=' SVM  Chi2 Bigrams Normalized confusion matrix')

plt.show()




    #Lsa 5
        SVM_big_lsa_5= train_predict_evaluate_model(lsa_features_big_tr,train_label,
                             lsa_features_big_te,test_label)
"""Export Data with pickle"""
with open('SVM_big_lsa_5', 'wb') as handle:
    pickle.dump(SVM_big_lsa_5, handle, protocol=pickle.HIGHEST_PROTOCOL)    

cnf_matrix = confusion_matrix(test_label, SVM_big_lsa_5)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title=' SVM Lsa Bow Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title=' SVM Lsa Bow Normalized confusion matrix')

plt.show()


"""
Bown
"""
    #F-classif
    
   SVM_bown_f= train_predict_evaluate_model(bown_tfidf_f_tr,train_label,
                             bown_tfidf_f_te,test_label)
"""Export Data with pickle"""
with open('SVM_bown_f', 'wb') as handle:
    pickle.dump(SVM_bown_f, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
     # Compute confusion matrix
cnf_matrix = confusion_matrix(test_label, SVM_bown_f)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title=' SVM Bow with nouns F-class Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title=' SVM Bow with nouns F-class Normalized confusion matrix')

plt.show()  

    
    #Chi2

    SVM_bown_chi2=train_predict_evaluate_model(bown_tfidf_chi2_tr,train_label,
                             bown_tfidf_chi2_te,test_label)

"""Export Data with pickle"""
with open('SVM_bown_chi2', 'wb') as handle:
    pickle.dump(SVM_bown_chi2, handle, protocol=pickle.HIGHEST_PROTOCOL)

 # Compute confusion matrix
cnf_matrix = confusion_matrix(test_label, SVM_bown_chi2)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title=' SVM Bow with nouns Chi2 Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title=' SVM Bow with nouns Chi2 Normalized confusion matrix')

plt.show()  


    #Lsa 5

        SVM_bown_lsa_5= train_predict_evaluate_model(lsa_features_bown_tr,train_label,
                             lsa_features_bown_te,test_label)
"""Export Data with pickle"""
with open('SVM_big_lsa_5', 'wb') as handle:
    pickle.dump(SVM_bown_lsa_5, handle, protocol=pickle.HIGHEST_PROTOCOL)    

cnf_matrix = confusion_matrix(test_label, SVM_bown_lsa_5)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title=' SVM Lsa Bow  with nouns Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title=' SVM Lsa Bow with nouns Normalized confusion matrix')

plt.show()





if __name__ == '__main__':
    main()