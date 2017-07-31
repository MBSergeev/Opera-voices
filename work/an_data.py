import essentia as es
from essentia.standard import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import train_test_split
#from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#import os

def find_seq(seq,dt=0.5):
   k_t=hopSize/float(sampleRate)
   isPitch=False
   delta_t=int(dt/k_t)
   acc_pitch=[]
   for i in range(len(seq)):
      f=seq[i]
      if not isPitch:
         if not np.isnan(f):
           f_start=f
           i_start=i
           isPitch=True
           pitch=get_score(f_start)
      else:
         if np.isnan(f):
           i_end=i#+1
           if i_end-i_start>delta_t:
              acc_pitch.append([i_start,i_end,pitch])
           isPitch=False
         else:
           if get_score(f)!=pitch:
              i_end=i#+1
              if i_end-i_start>delta_t:
                 acc_pitch.append([i_start,i_end,pitch])
              f_start=f
              pitch=get_score(f_start)
              i_start=i
 
   if isPitch:
      i_end=i#+1
      if i_end-i_start>delta_t:
         acc_pitch.append([i_start,i_end,pitch])
   return acc_pitch

def find_seq_s(seq,dt=0.5):
   k_t=hopSize/float(sampleRate)
   isPitch=False
   delta_t=int(dt/k_t)
   acc_pitch=[]
   for i in range(len(seq)):
      f=seq[i]
      if not isPitch:
         if not np.isnan(f):
           f_start=f
           i_start=i
           isPitch=True
      else:
         if np.isnan(f):
           i_end=i
           if i_end-i_start>delta_t:
              acc_pitch.append([i_start,i_end])
           isPitch=False
 
   if isPitch:
      i_end=i
      if i_end-i_start>delta_t:
         acc_pitch.append([i_start,i_end])
   return acc_pitch

def get_score(fr_curr):
    if not fr_curr or np.isnan(fr_curr):
        return (None,np.nan)
    scores0=['c','c#','d','d#','e','f','f#','g','g#','a','a#','h']
    freq=[]
    for i in range(3,12*5+3):
        fr=55*(2)**(i/12.0)        
        freq.append(fr)
    
    scores=[]
    
    for x in scores0:
        scores.append(x+'-mag')
    for x in scores0:
        scores.append(x+'-min')        
    for x in scores0:
        scores.append(x+'-1')        
    for x in scores0:
        scores.append(x+'-2')  
    for x in scores0:
        scores.append(x+'-3')      
    
    min_d=99999999999
    for fr in zip(scores,freq):
        delta=abs(fr[1]-fr_curr)
        if delta<min_d:
            min_d=delta
            min_f=fr
    return min_f


def smooth(x,window_len=11,window='hanning'):
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
   
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
           
   
    if window_len<3:
        return x
       
       
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
       
   
    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
  
    if window == 'flat': #moving average
          w=np.ones(window_len,'d')
    else:
          w=eval('np.'+window+'(window_len)')
       
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[(window_len/2):-(window_len/2)]



def an_data_learn():
   t0=49
   t1=72

   data=pd.read_table(dir_data+"OP_Se_in_ciel.dat")

   time=data.loc[:,"Time"].as_matrix()
   freq=data.loc[:,"Frequency"].as_matrix()
   pitch=data.loc[:,"Pitch"].as_matrix()

   pitch[pitch==0]=np.nan
   freq[freq==0]=np.nan
   f0_sm=smooth(freq,win_len,window='hamming')

   mfcc_lst=[]
   for i in range(numberMFCC):
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
#  scaler=StandardScaler()
#   scaler.fit(X_train)
#   X_train = scaler.transform(X_train)
#   X_test = scaler.transform(X_test)
   clf=linear_model.LogisticRegression(C=0.1,random_state=261,penalty='l1')
   clf.fit(X_train,y_train)
   y_pred=clf.predict(X_test)
   print "log-regression MFCC1",accuracy_score(y_test,y_pred)
   return clf
    
def an_data_use(Filename,clf):
   print Filename
   data=pd.read_table(dir_data+Filename+".dat")
   time=data.loc[:,"Time"].as_matrix()
   freq=data.loc[:,"Frequency"].as_matrix()
   pitch=data.loc[:,"Pitch"].as_matrix()
   pitch[pitch==0]=np.nan
   freq[freq==0]=np.nan
   f0_sm=smooth(freq,win_len,window='hamming')

   mfcc_lst=[]
   for i in range(numberMFCC):
      mfcc_lst.append("MFCC1_"+str(i))

   harm_lst=[]
   for i in range(nHarmonics):
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
   for i in range(nHarmonics):
      harm_tbl_sm[:,i]=smooth(harm_tbl[:,i],win_len,window='hamming')

   mfcc_tbl_sm=np.empty(mfcc_tbl.shape)
   for i in range(numberMFCC):
      mfcc_tbl_sm[:,i]=smooth(mfcc_tbl[:,i],win_len,window='hamming')

#   mfcc_tbl_sing=np.empty((0,numberMFCC))
#   f0_sm_sing=[]

   f0_sm_m_l=[]
   harm_m_l=np.empty((0,nHarmonics))
   mfcc_m_l=np.empty((0,numberMFCC))
   cnt=0
   for seq in find_seq(f0_sm,0.5):
         #print "cnt=",cnt
#         t0=time[seq[0]]
#         t1=time[seq[1]-1]
#         plt.plot(time[seq[0]:seq[1]],freq[seq[0]:seq[1]],color='b')
#         plt.plot(time[seq[0]:seq[1]],f0_sm[seq[0]:seq[1]],color='cyan')
         """
         for i in range(1,nHarmonics):
            if i==0:
                clr='black'
            elif i==1:
                clr='red'
            elif i==2:
                clr='orange'
            elif i==3:
                clr='yellow'
            elif i==4:
                clr='green'
            elif i==5:
                clr='lightblue'
            elif i==6:
                clr='blue'
            elif i==7:
                clr='violet'
            elif i==8:
                clr='cyan'
            else:
                clr='black'
# harmonics on time
#            plt.plot(time[seq[0]:seq[1]],harm_tbl_sm[seq[0]:seq[1],i]-harm_tbl_sm[seq[0]:seq[1],0],color=clr)
# mfcc-coef on time
#            plt.plot(time[seq[0]:seq[1]],mfcc_tbl_sm[seq[0]:seq[1],i],color=clr)
         """
#  all spectra for a tone
#         for j in range(seq[0],seq[1]):
#             plt.plot((np.arange(nHarmonics)+1)*f0_sm[j],harm_tbl_sm[j,:]-harm_tbl_sm[j,0])
             
         f0_sm_m=np.mean(f0_sm[seq[0]:seq[1]-1])
         f0_sm_m_l=np.append(f0_sm_m_l,f0_sm_m)
         
         harm_m = np.mean(harm_tbl_sm[seq[0]:seq[1],:] -  harm_tbl_sm[seq[0]:seq[1],0].reshape(seq[1]-seq[0],1),axis=0)
         harm_m_l=np.append(harm_m_l,[harm_m],axis=0)

         mfcc_m = np.mean(mfcc_tbl_sm[seq[0]:seq[1],:],axis=0)
         mfcc_m_l=np.append(mfcc_m_l,[mfcc_m],axis=0)
#         plt.plot(np.arange(numberMFCC-1)+2,mfcc_m[1:])

# middle spectrum
#         plt.plot((np.arange(nHarmonics)+1)*f0_sm_m,harm_m,color=col)

#         plt.plot([time[seq[0]],time[seq[0]-t0]],[0,1000],color="g")
#         plt.plot([time[seq[1]],time[seq[1]]],[0,1000],color="r")
# the syntesis of sound
#         syn_sound(data.loc[seq[0]:seq[1]-1,:],"sounds/"+"%02d" % cnt+"_"+Filename+".wav")

#         mfcc_tbl_sing=np.vstack([mfcc_tbl_sing,mfcc_tbl_sm[seq[0]:seq[1],:]]) 
#         f0_sm_sing=np.append(f0_sm_sing,f0_sm[seq[0]:seq[1]])
         cnt+=1

#   plt.xlim((0,5000))
#   plt.show()
#   plt.hist(f0_sm_m_l)
#   plt.show()

#   regr = linear_model.LinearRegression()
#   X_train, X_test, y_train, y_test = train_test_split(mfcc_tbl_sing[:,1:10], f0_sm_sing, test_size=0.33, random_state=241)
#   regr.fit(X_train,y_train)

#   print "Error=",np.sqrt(np.mean((regr.predict(X_test) - y_test) ** 2))
#   plt.scatter(regr.predict(X_test),y_test)
#   plt.show()
   return f0_sm_m_l,harm_m_l,mfcc_m_l

#Mean spectra in bark-intervals
def mean_spectra(tr_lst,clf):
  harm_m=np.zeros((nBarkband,nHarmonics))
  cnt_m=np.zeros(nBarkband)
  mfcc_m=np.zeros((nBarkband,numberMFCC))
  for tr in tr_lst:
    f0,harm,mfcc = an_data_use(tr,clf)
#    print np.min(f0),np.max(f0)
    for i in range(len(f0)): 
        for i_m in range(len(marg)):
           if marg[i_m]>f0[i]:
              break
   

        harm_s=harm_m[i_m-1]
        harm_s+=harm[i]
        harm_m[i_m-1]=harm_s

        mfcc_s=mfcc_m[i_m-1]
        mfcc_s+=mfcc[i]
        mfcc_m[i_m-1]=mfcc_s

        cnt_m[i_m-1]+=1


        if i_m==5:
            clr='red'
        elif i_m==6:
            clr='orange'
        elif i_m==7:
            clr='yellow'
        elif i_m==8:
            clr='green'
        elif i_m==9:
            clr='lightblue'
        elif i_m==10:
            clr='blue'
        elif i_m==11:
            clr='violet'
        else:
            clr='black'
  print "--------"
  for i_m in range(5,nBarkband):
        if i_m==5:
            clr='red'
        elif i_m==6:
            clr='orange'
        elif i_m==7:
            clr='yellow'
        elif i_m==8:
            clr='green'
        elif i_m==9:
            clr='lightblue'
        elif i_m==10:
            clr='blue'
        elif i_m==11:
            clr='violet'
        else:
            clr='black'
        if cnt_m[i_m]>0:
            print i_m,clr,marg_m[i_m],int(cnt_m[i_m]),get_score(marg_m[i_m])[0]
            harm_m[i_m]=harm_m[i_m]/cnt_m[i_m]
            plt.plot((np.arange(nHarmonics)+1)*marg_m[i_m],harm_m[i_m],color=clr)

  plt.xlim((0,5000))
  plt.show()

  for i_m in range(5,nBarkband):
        if i_m==5:
            clr='red'
        elif i_m==6:
            clr='orange'
        elif i_m==7:
            clr='yellow'
        elif i_m==8:
            clr='green'
        elif i_m==9:
            clr='lightblue'
        elif i_m==10:
            clr='blue'
        elif i_m==11:
            clr='violet'
        else:
            clr='black'
        if cnt_m[i_m]>0:
            mfcc_m[i_m]=mfcc_m[i_m]/cnt_m[i_m]
            plt.plot(np.arange(numberMFCC-1)+2,mfcc_m[i_m,1:],color=clr)

  plt.show()


nHarmonics=15
frameSize = 2*1024
hopSize = 256 
sampleRate = 44100
nBarkband=27
numberMFCC=16
marg=[0,50,100,150,200,300,400,510,630,770,920,1080,1270,1480,1720,2000,2320,2700,3150,3700,4400,5300,6400,7700,9500,12000,15500,20500]#,27000]
dir_data="../data/"

marg_m=[] 
marg_d=[] 
for i in range(nBarkband):
  marg_m=np.append(marg_m,(marg[i]+marg[i+1])/2.0)
  marg_d=np.append(marg_d,marg[i+1]-marg[i])

k_t=hopSize/float(sampleRate)
win_len=199


clf=an_data_learn()

tr_lst=["OP_Se_in_ciel","OP_Crudele","OP_Bacio","OP_Deh_Vieni","OP_Ah_non_potrian","OP_Merce_dilette","OP_Nachtigall","OP_Son_Vergin",
   "OP_O_legere_hirondelle","OP_Spiel_ich","OP_O_Rendetemi","OP_Villanelle","OP_Ouvre_ton_coer"]

mean_spectra(tr_lst,clf)

