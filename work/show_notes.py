# -*- coding: utf-8 -*-
# показать для одного трека выделенные ноты
import essentia as es
from essentia.standard import *

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from my_cons import my_cons

def an_data_learn(Filename="OP_Se_in_ciel.dat",t0=49,t1=72):

   data=pd.read_table(my_c.dir_data+Filename)

   time=data.loc[:,"Time"].as_matrix()
   freq=data.loc[:,"Frequency"].as_matrix()
   pitch=data.loc[:,"Pitch"].as_matrix()

   pitch[pitch==0]=np.nan
   freq[freq==0]=np.nan
   f0_sm=smooth(freq,my_c.win_len,window='hamming')

   mfcc_lst=[]
   for i in range(my_c.numberMFCC):
      mfcc_lst.append("MFCC1_"+str(i))

   mfcc_tbl=data.loc[:,mfcc_lst].as_matrix()
   X=mfcc_tbl[time<t1]
   X=X[:,1:]

   y=np.zeros(X.shape[0])

   time1=time[time<t1]
   y[time1>t0]=1


   fr=freq[time<t1]
   X=X[~np.isnan(fr)]
   y=y[~np.isnan(fr)]

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=261)
   clf=linear_model.LogisticRegression(C=0.1,random_state=261,penalty='l1')
   clf.fit(X_train,y_train)
   y_pred=clf.predict(X_test)
   print "log-regression MFCC1",accuracy_score(y_test,y_pred)
   return clf


def an_data(Filename,clf):
   print Filename
   data=pd.read_table(my_c.dir_data+Filename+".dat")
   time=data.loc[:,"Time"].as_matrix()
   freq=data.loc[:,"Frequency"].as_matrix()
   pitch=data.loc[:,"Pitch"].as_matrix()
   pitch[pitch==0]=np.nan
   freq[freq==0]=np.nan
   f0_sm=smooth(freq,my_c.win_len,window='hamming')

   mfcc_lst=[]
   for i in range(my_c.numberMFCC):
      mfcc_lst.append("MFCC1_"+str(i))

   harm_lst=[]
   for i in range(my_c.nHarmonics):
      harm_lst.append("Harm_"+str(i))

   mfcc_tbl=data.loc[:,mfcc_lst].as_matrix()
   mfcc_tbl=mfcc_tbl[~np.isnan(freq)]

   mfcc_tbl=mfcc_tbl[:,1:]
   cl=clf.predict(mfcc_tbl)

   cl_n=np.zeros(freq.shape[0])*np.nan
   j=0
   for i in range(cl_n.shape[0]):
      if not np.isnan(freq[i]):
          cl_n[i]=cl[j]
          j+=1

   f0_sm[cl_n==0]=np.nan

   mfcc_tbl=data.loc[:,mfcc_lst].as_matrix()
   harm_tbl=data.loc[:,harm_lst].as_matrix()

   harm_tbl_sm=np.empty(harm_tbl.shape)
   for i in range(my_c.nHarmonics):
      harm_tbl_sm[:,i]=smooth(harm_tbl[:,i],my_c.win_len,window='hamming')

   mfcc_tbl_sm=np.empty(mfcc_tbl.shape)
   for i in range(my_c.numberMFCC):
      mfcc_tbl_sm[:,i]=smooth(mfcc_tbl[:,i],my_c.win_len,window='hamming')



   cnt=0 
   t_a=0.0
   for seq in find_seq(f0_sm,0.5):
         t0=time[seq[0]]
         t1=time[seq[1]-1]
#         """ 
         for i in range(-12,16):
             fr_t=2**(i/12.0)*440
             plt.plot(time[seq[0]:seq[1]-1]-t0+t_a,fr_t*np.ones(seq[1]-seq[0]-1),'black')
             plt.text(0,fr_t,get_score(fr_t)[0])

         plt.plot(time[seq[0]:seq[1]-1]-t0+t_a,freq[seq[0]:seq[1]-1],color='b')
         plt.plot(time[seq[0]:seq[1]-1]-t0+t_a,f0_sm[seq[0]:seq[1]-1],color='cyan')
#         """
#         plt.plot(time[seq[0]:seq[1]-1]-t0+t_a,freq[seq[0]:seq[1]-1]/10,color='cyan')
#         plt.plot(time[seq[0]:seq[1]-1]-t0+t_a,f0_sm[seq[0]:seq[1]-1]/10,color='cyan')
 #        plt.plot(time[seq[0]:seq[1]-1]-t0+t_a,mfcc_tbl[seq[0]:seq[1]-1,1],color='b')
#         plt.plot(time[seq[0]:seq[1]-1]-t0+t_a,mfcc_tbl_sm[seq[0]:seq[1]-1,1],color='b')
 #        plt.plot(time[seq[0]:seq[1]-1]-t0+t_a,mfcc_tbl[seq[0]:seq[1]-1,2],color='g')
#         plt.plot(time[seq[0]:seq[1]-1]-t0+t_a,mfcc_tbl_sm[seq[0]:seq[1]-1,2],color='g')
         
         t_a=t_a+t1-t0
         cnt+=1
   plt.show() 

my_c=my_cons()

clf=an_data_learn()

an_data("OP_Deh_Vieni",clf)

