
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


lst=[]
#lst=["f0_sm"]
for i in range(5,16):
   lst.append("MFCC1_"+str(i))

X=data.loc[:,lst].as_matrix()[(300<f0_sm) & (f0_sm<1000)]
y= np.where(Singer=="OP",1,0)[(300<f0_sm) & (f0_sm<1000)]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=261)
clf=linear_model.LogisticRegression(C=0.1,random_state=261,penalty='l2')
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print "log-regression MFCC1",roc_auc_score(y_test,y_pred)

pca = PCA(n_components=2)
pca.fit(X)
X_tr=pca.transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_tr, y, test_size=0.2, random_state=261)
clf1=linear_model.LogisticRegression(C=0.1,random_state=261,penalty='l2')
clf1.fit(X_train,y_train)
y_pred=clf1.predict(X_test)
print "log-regression MFCC1 PCA",roc_auc_score(y_test,y_pred)
