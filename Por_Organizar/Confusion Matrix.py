#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 10:57:57 2018

@author: MarioAntao
"""

-----------
import pandas as pd
cm = metrics.confusion_matrix(test_label, lsa_bow)
pd.DataFrame(cm, index=range(0,4), columns=range(0,4))  


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def cm_analysis(y_true, y_pred, labels, ymap=None, figsize=(10,10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap != None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    plt.show()

cm_analysis(test_label, SVM_bow_lsa_5, class_names, ymap=None, figsize=(10,10))

------

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

class_names=['Baby','Cd and Vinyl','Digital Music','Toys and Games']



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(test_label, SVM_bow_lsa_5)

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title=' SVM Bow Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title=' SVM Bow Normalized confusion matrix')

plt.show()




  matplotlib.figure.Figure
        The resulting confusion matrix figure
    
    df_cm = pd.DataFrame(
        cnf_matrix, index=class_names, columns=class_names
    )
    fig = plt.figure(figsize=(10,10))

df_norm_col=(df_cm-df_cm.mean())/df_cm.std()
sns.heatmap(df_norm_col, cmap='viridis')
#sns.plt.show()

heawtmap = sns.heatmap(df_norm_col, annot=True,square=True).set_title('SVM with LSA Bow')



from sklearn.decomposition import LatentDirichletAllocation

import seaborn as sns
import matplotlib.pyplot as plt

a=random.sample(train_corpus, 100)
def display_heatmap(df, cmap='Blues', x=10, y=8):
    fig, ax = plt.subplots(figsize=(x, y))   
    return sns.heatmap(df, cmap=cmap, annot=True, mask=df.isnull(), ax=ax)
def display_lda_doc_topic(lda_features, index):
    return pd.DataFrame(lda_features, index=index)
     
lda_model = LatentDirichletAllocation(n_components=4, doc_topic_prior=0.5, topic_word_prior=0.5)
lda_features = lda_model.fit_transform(bow_tr_idfs)
plt.savefig('Plot TF_IDF'+str(display_heatmap(display_lda_doc_topic(lda_features, a), y = 50))+'.jpg')







