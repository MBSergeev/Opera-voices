import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

data=pd.read_table("Adina_mfcc.dat")
sng_lst= ["OP","AN","VN","LP","MD","JS","AG","AK","LC","AF","IC","KB","ES","RS"]

for sng in sng_lst:
   if sng=="OP":
       col="b"
   elif sng=="AN":
       col="r"
   elif sng=="VN":
       col="g"
   elif sng=="LP":
       col="orange"
   elif sng=="MD":
       col="magenta"
   elif sng=="JS":
       col="cyan"
   elif sng=="AG":
       col="black"
   elif sng=="AK":
       col="yellow"
   elif sng=="LC":
       col="lightblue"
   elif sng=="AF":
       col="lightgreen"
   elif sng=="IC":
       col="g"
   elif sng=="KB":
       col="cyan"
   elif sng=="ES":
       col="b"
   elif sng=="RS":
       col="black"
   data_sng=data[data.loc[:,'Singer']==sng]
   plt.scatter(data_sng.loc[:,'MFCC1_2'],data_sng.loc[:,'MFCC1_3'],color=col)
#plt.show()


cont1=[]
cont2=[]

for i in range(2,8):
   for j in range(i+1,8):
      cont1=np.append(cont1,np.corrcoef(data_sng.loc[:,'MFCC1_'+str(i)],data_sng.loc[:,'MFCC1_'+str(j)])[0,1])
      cont2.append([i,j])
#      plt.scatter(data_sng.loc[:,'Harm_p_'+str(i)],data_sng.loc[:,'Harm_p_'+str(j)],color=col)
#      plt.show()

for i in  np.argsort(cont1):
   print cont2[i],cont1[i]

sys.exit(0)

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

n_f=4
kf=KFold(n_splits=n_f,shuffle=True,random_state=242)

lst0=[]
#for k in range(2,16):
for k in range(2,9):
   lst0.append('MFCC1_'+str(k))

lst=[]
for n in range(6,7):
#   lst.append(lst0[n])
   lst=lst0
  

   X=data.loc[:,lst]
   y=data.loc[:,'Singer']

   m_score_max=0
   for n_b in range(2,20):

      s_score=0
      for train_index,test_index in kf.split(X):
         X_train,X_test =X.iloc[train_index],X.iloc[test_index]
         y_train,y_test =y.iloc[train_index],y.iloc[test_index]

         neigh=KNeighborsClassifier(n_neighbors=n_b)
         neigh.fit(X_train,y_train)
         sc = neigh.score(X_test,y_test)
#      print "score=",sc
         s_score+=sc

#      print "n_neigbors=",n_b,"Mean score=",s_score/n_f
      if s_score/n_f>m_score_max:
          m_score_max=s_score/n_f
          n_b_max=n_b
   print  n,"n_b_max=",n_b_max,"m_score_max=",m_score_max

from sklearn.metrics import accuracy_score
OP=data[data.loc[:,'Singer']=='OP']
AN=data[data.loc[:,'Singer']=='AN']
data_n=pd.concat([OP,AN])

X=data_n.loc[:,lst]
y=data_n.loc[:,'Singer']

m_score_max=0
for n_b in range(2,20):

  s_score=0
  for train_index,test_index in kf.split(X):
      X_train,X_test =X.iloc[train_index],X.iloc[test_index]
      y_train,y_test =y.iloc[train_index],y.iloc[test_index]

      neigh=KNeighborsClassifier(n_neighbors=n_b)
      neigh.fit(X_train,y_train)
      sc = neigh.score(X_test,y_test)
      predictions=  neigh.predict(X_test)
      ac= accuracy_score(y_test,predictions)
#      print "score=",sc
      s_score+=ac

#  print "n_neigbors=",n_b,"Mean score=",s_score/n_f
  if s_score/n_f>m_score_max:
          m_score_max=s_score/n_f
          n_b_max=n_b
print  n,"n_b_max=",n_b_max,"m_score_max=",m_score_max

from sklearn.linear_model import Perceptron

kf=KFold(n_splits=n_f,shuffle=True,random_state=242)
sum_ac=0
for train_index,test_index in kf.split(X):
    X_train,X_test =X.iloc[train_index],X.iloc[test_index]
    y_train,y_test =y.iloc[train_index],y.iloc[test_index]
    clf = Perceptron(random_state=242)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    ac= accuracy_score(y_test,predictions)
    sum_ac+=ac
print "Perceptron Mid ac=",sum_ac/n_f

from sklearn import svm


kf=KFold(n_splits=n_f,shuffle=True,random_state=242)
sum_ac=0
for train_index,test_index in kf.split(X):
    X_train,X_test =X.iloc[train_index],X.iloc[test_index]
    y_train,y_test =y.iloc[train_index],y.iloc[test_index]
    clf = svm.SVC(kernel='linear',random_state=242)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    ac= accuracy_score(y_test,predictions)
    sum_ac+=ac
print "svm Mid ac=",sum_ac/n_f


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
kf=KFold(n_splits=n_f,shuffle=True,random_state=242)
sum_ac=0
for train_index,test_index in kf.split(X):
    X_train,X_test =X.iloc[train_index],X.iloc[test_index]
    y_train,y_test =y.iloc[train_index],y.iloc[test_index]
    clf=GradientBoostingClassifier(n_estimators=20,random_state=241)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    ac= accuracy_score(y_test,y_pred)
    sum_ac+=ac
print "GradientBoostingClassifier Mid ac=",sum_ac/n_f

lst=['Mean_frequency']
for k in range(2,8):
   lst.append('MFCC1_'+str(k))
X=data_n.loc[:,lst]
y=data_n.loc[:,'Singer']

kf=KFold(n_splits=n_f,shuffle=True,random_state=242)
sum_ac=0
for train_index,test_index in kf.split(X):
    X_train,X_test =X.iloc[train_index],X.iloc[test_index]
    y_train,y_test =y.iloc[train_index],y.iloc[test_index]
    clf=GradientBoostingClassifier(n_estimators=25,random_state=241)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    ac= accuracy_score(y_test,y_pred)
    sum_ac+=ac
print "GradientBoostingClassifier Mid ac=",sum_ac/n_f
