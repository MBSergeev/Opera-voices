# -*- coding: utf-8 -*-
# для длинных нот получить файл с данными - частота - MFCC
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

   fp=open("res.dat","a")
   cnt=0 
   t_a=0.0
   for seq in find_seq(f0_sm,0.5):
         t0=time[seq[0]]
         t1=time[seq[1]-1]
#         """ 
#         for i in range(-12,16):
#             fr_t=2**(i/12.0)*440
#             plt.plot(time[seq[0]:seq[1]-1]-t0+t_a,fr_t*np.ones(seq[1]-seq[0]-1),'black')

#         plt.plot([t_a,t_a],[-90,80],'g') 


#         plt.plot(time[seq[0]:seq[1]-1]-t0+t_a,freq[seq[0]:seq[1]-1],color='b')
#         plt.plot(time[seq[0]:seq[1]-1]-t0+t_a,f0_sm[seq[0]:seq[1]-1],color='cyan')
#         """
#         plt.plot(time[seq[0]:seq[1]-1]-t0+t_a,freq[seq[0]:seq[1]-1]/10,color='cyan')
#         plt.plot(time[seq[0]:seq[1]-1]-t0+t_a,f0_sm[seq[0]:seq[1]-1]/10,color='cyan')
#          plt.plot(time[seq[0]:seq[1]-1]-t0+t_a,mfcc_tbl[seq[0]:seq[1]-1,1],color='b')
#         plt.plot(time[seq[0]:seq[1]-1]-t0+t_a,mfcc_tbl_sm[seq[0]:seq[1]-1,1],color='r')
#          plt.plot(time[seq[0]:seq[1]-1]-t0+t_a,mfcc_tbl[seq[0]:seq[1]-1,2],color='g')
#         plt.plot(time[seq[0]:seq[1]-1]-t0+t_a,mfcc_tbl_sm[seq[0]:seq[1]-1,2],color='orange')
#         plt.plot(time[seq[0]:seq[1]-1]-t0+t_a,mfcc_tbl_sm[seq[0]:seq[1]-1,3],color='yellow')
#         plt.plot(time[seq[0]:seq[1]-1]-t0+t_a,mfcc_tbl_sm[seq[0]:seq[1]-1,4],color='green')
         
         t_a=t_a+t1-t0
         f0_m=f0_sm[seq[0]:seq[1]-1].mean()
         fp.write(Filename+"\t"+str(cnt)+"\t"+str(f0_m))
         for i_mfc in range(my_c.numberMFCC):
             mfcc_m=mfcc_tbl_sm[seq[0]:seq[1]-1,i_mfc].mean()
             fp.write("\t"+str(mfcc_m))
         fp.write("\n")
         cnt+=1
#   plt.show() 

   fp.close()

my_c=my_cons()

clf=an_data_learn()
fp=open("res.dat","w")
fp.write("Filename\tcnt\tf0_sm")
for i_mfc in range(my_c.numberMFCC):
    fp.write("\tMFCC1_"+str(i_mfc))
fp.write("\n")
fp.close()

an_data("AN_Addio_del_passato",clf)
#an_data("AN_Adina",clf)
an_data("AN_Ardon_gli_incensi",clf)
an_data("AN_Deh_Vieni",clf)
an_data("AN_Giuditta",clf)
an_data("AN_Lauretta",clf)
an_data("AN_Mimi",clf)
an_data("AN_Musetta",clf)
an_data("AN_Norma",clf)
an_data("AN_Oh_quante_volte",clf)
an_data("AN_Rusalka",clf)
an_data("AN_Solvejgs",clf)
#an_data("OP_Adina",clf)
an_data("OP_Ah_non_potrian",clf)
an_data("OP_All_ombra",clf)
an_data("OP_Amenaide",clf)
an_data("OP_Bacio",clf)
an_data("OP_Bel_raggio",clf)
an_data("OP_Caro_nome",clf)
an_data("OP_Crudele",clf)
an_data("OP_Deh_Vieni",clf)
an_data("OP_Doretta",clf)
an_data("OP_Eshchyo_v_polyakh",clf)
an_data("OP_I_Vostri",clf)
an_data("OP_Lidochka1",clf)
an_data("OP_Lidochka2",clf)
an_data("OP_Linda",clf)
an_data("OP_Ludmila",clf)
an_data("OP_Manon",clf)
an_data("OP_Matilde",clf)
an_data("OP_Mein_Herr_Marquis",clf)
an_data("OP_Merce_dilette",clf)
an_data("OP_Nachtigall",clf)
an_data("OP_Ne_poy",clf)
an_data("OP_Nightingale",clf)
an_data("OP_Non_si_da",clf)
an_data("OP_Norina",clf)
an_data("OP_O_legere_hirondelle",clf)
an_data("OP_Olympia",clf)
an_data("OP_O_Rendetemi",clf)
an_data("OP_Otello",clf)
an_data("OP_Ouvre_ton_coer",clf)
an_data("OP_Partir",clf)
an_data("OP_Plenivshis_rozoj",clf)
an_data("OP_Prendi_per_me",clf)
an_data("OP_Regnava_nel_silenzio",clf)
an_data("OP_Rosina",clf)
an_data("OP_Rusalka",clf)
an_data("OP_Se_in_ciel",clf)
an_data("OP_Shemahanskaya_zarica",clf)
an_data("OP_Snegurochka",clf)
an_data("OP_Son_Vergin",clf)
an_data("OP_Spiel_ich",clf)
an_data("OP_Villanelle",clf)
an_data("OP_Vocalise",clf)
an_data("OP_Volhova",clf)
an_data("OP_Zarskaya_nevesta",clf)
an_data("OP_Zdes_khorosho",clf)

