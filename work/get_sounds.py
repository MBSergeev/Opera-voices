import essentia as es
from essentia.standard import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score

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

def syn_sound(data,name):
   sinemodelsynth = SineModelSynth(sampleRate=sampleRate,fftSize=frameSize,hopSize=hopSize)
   ifft=IFFT(size=frameSize)
   overlapAdd = OverlapAdd(frameSize=frameSize, hopSize=hopSize,  gain=1./frameSize)
   audioWriter = MonoWriter(filename=name)
   audioout=[]
   magn_lst=[]
   phase_lst=[]

   for i in range(nHarmonics):
      magn_lst=np.append(magn_lst,"Harm_"+str(i))
      phase_lst=np.append(magn_lst,"Phase_"+str(i))

   for cnt in range(data.shape[0]):
       dt=data.iloc[cnt,:]
       magn=dt[magn_lst]
       phase=dt[phase_lst]
       freq=[]
       for j in range(nHarmonics):
          fr0=dt['Frequency']
          freq=np.append(freq,fr0*(j+1))
       sfftframe=sinemodelsynth(es.array(magn),es.array(freq),es.array(phase))
       ifftframe=ifft(sfftframe)
       out=overlapAdd(ifftframe)
       audioout = np.append(audioout,out)

# Write to file
   audioWriter(audioout.astype(np.float32))
                                              

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
      syn_sound(data.loc[seq[0]:seq[1]-1,:],dir_out_sound+Filename+"_"+"%02d" % cnt+".wav")
      cnt+=1



dir_data="../data/"
dir_out_sound="../out_sound/"
win_len=199
numberMFCC=16
hopSize = 256
sampleRate = 44100
frameSize = 2*1024
nHarmonics=15



clf=an_data_learn()
an_data_use("OP_Se_in_ciel",clf)

