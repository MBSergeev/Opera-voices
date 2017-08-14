import essentia as es
from essentia.standard import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score

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

def an_data_use(Filename,clf,all_data=False):
   print Filename
   data=pd.read_table(my_c.dir_data+Filename+".dat")

   if all_data:
        syn_sound(data,my_c.dir_out_sound+Filename+"_out.wav")
        return


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

   cnt=0
   for seq in find_seq(f0_sm,0.5):
      print cnt
      syn_sound(data.loc[seq[0]:seq[1]-1,:],my_c.dir_out_sound+Filename+"_"+"%02d" % cnt+".wav")
      cnt+=1



my_c=my_cons()


#clf=an_data_learn("OP_Regnava_nel_silenzio.dat",21,36)
clf=an_data_learn()
#an_data_use("OP_Se_in_ciel",clf)
#an_data_use("OP_Se_in_ciel",clf,True)
# Write all sound
#an_data_use("OP_Ludmila",clf)
#an_data_use("OP_Ludmila",clf,all_data=True)
#an_data_use("OP_Shemahanskaya_zarica",clf,all_data=True)
#an_data_use("OP_Shemahanskaya_zarica",clf)
#an_data_use("OP_Snegurochka",clf,all_data=True)
#an_data_use("OP_Snegurochka",clf)
#an_data_use("OP_Volhova",clf)
#an_data_use("OP_Zarskaya_nevesta",clf)
#an_data_use("OP_Plenivshis_rozoj",clf)
#an_data_use("OP_Eshchyo_v_polyakh",clf)
#an_data_use("OP_Vocalise",clf)
#an_data_use("OP_Ne_poy",clf)
#an_data_use("OP_Zdes_khorosho",clf)
#an_data_use("OP_Nightingale",clf)
#an_data_use("OP_Lidochka1",clf)
#an_data_use("OP_Lidochka2",clf)
#an_data_use("OP_Adina",clf)
#an_data_use("AN_Adina",clf)

#an_data_use("OP_Olympia",clf)
#an_data_use("OP_Caro_nome",clf)
#an_data_use("OP_Rusalka",clf)
#an_data_use("OP_Doretta",clf)
#an_data_use("OP_Otello",clf)
#an_data_use("OP_Manon",clf)
#an_data_use("OP_Linda",clf)
#an_data_use("OP_Prendi_per_me",clf)
#an_data_use("OP_Norina",clf)
#an_data_use("OP_Regnava_nel_silenzio",clf)

#an_data_use("AN_Lauretta",clf)
#an_data_use("AN_Musetta",clf)
#an_data_use("AN_Deh_Vieni",clf)
#an_data_use("AN_Ardon_gli_incensi",clf)
#an_data_use("AN_Norma",clf)
#an_data_use("AN_Rusalka",clf)
#an_data_use("AN_Giuditta",clf)
an_data_use("AN_Mimi",clf)

