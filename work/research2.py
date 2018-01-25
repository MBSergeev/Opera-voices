
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_table("res.dat")

Filename=data.loc[:,"Filename"].as_matrix()
f0_sm=data.loc[:,"f0_sm"].as_matrix()

np.set_printoptions(threshold='nan')

Singer=[]
for fn in Filename:
   Singer=np.append(Singer,fn[0:2])
"""
for i_step in range(0,16**4,2):
   ss="{0:016b}".format(i_step)[::-1]
   lst=[]
   for i in range(len(ss)):
      if ss[i]=="1": 
         lst.append("MFCC1_"+str(i))
   if len(lst)==0:
     continue
    
   X=data.loc[:,lst].as_matrix()[(300<f0_sm) & (f0_sm<1000)]
   y= np.where(Singer=="OP",1,0)[(300<f0_sm) & (f0_sm<1000)]
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=261)
   clf=linear_model.LogisticRegression(C=0.1,random_state=261,penalty='l1')
   clf.fit(X_train,y_train)
   y_pred=clf.predict(X_test)
   print "log-regression MFCC1",i_step,lst,roc_auc_score(y_test,y_pred)
#  print clf.coef_
   pca = PCA(n_components=i_data)
   pca.fit(X)
   X_tr=pca.transform(X)


   X_train, X_test, y_train, y_test = train_test_split(X_tr, y, test_size=0.2, random_state=261)
   clf1=linear_model.LogisticRegression(C=0.1,random_state=261,penalty='l1')
   clf1.fit(X_train,y_train)
   y_pred=clf1.predict(X_test)
#   print "log-regression MFCC1 PCA",i_data,roc_auc_score(y_test,y_pred)
#   print clf.coef_
"""
lst=['MFCC1_2','MFCC1_5','MFCC1_7','MFCC1_8','MFCC1_9','MFCC1_10','MFCC1_12']
X=data.loc[:,lst].as_matrix()[(300<f0_sm) & (f0_sm<1000)]
y= np.where(Singer=="OP",1,0)[(300<f0_sm) & (f0_sm<1000)]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=261)
clf=linear_model.LogisticRegression(C=0.1,random_state=261,penalty='l1')
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print "log-regression MFCC1",lst,roc_auc_score(y_test,y_pred)

pca = PCA(n_components=7)
pca.fit(X)
X_tr=pca.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_tr, y, test_size=0.2, random_state=261)
clf1=linear_model.LogisticRegression(C=0.1,random_state=261,penalty='l1')
clf1.fit(X_train,y_train)
y_pred=clf1.predict(X_test)
print "log-regression MFCC1 PCA",roc_auc_score(y_test,y_pred)

