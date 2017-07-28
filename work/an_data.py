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
           i_end=i+1
           if i_end-i_start>delta_t:
              acc_pitch.append([i_start,i_end])
           isPitch=False
 
   if isPitch:
      i_end=i+1
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


def an_data_test(Filename):
   data=pd.read_table(Filenaame)
   data_sel=data[data.Filename==Filename]
   time=data_sel.loc[:,"Time"].as_matrix()
   freq=data_sel.loc[:,"Frequency"].as_matrix()
   pitch=data_sel.loc[:,"Pitch"].as_matrix()
   pitch[pitch==0]=np.nan
   freq[freq==0]=np.nan
   f0_sm=smooth(freq,win_len,window='hamming')
   t0=49
   t1=72
   """ 
   plt.plot(time,freq)
   plt.plot(time,f0_sm)
   for i in range(-12,16):
      fr_t=2**(i/12.0)*440
      plt.plot(time,fr_t*np.ones(time.size),'black')
      plt.text(t0,fr_t,get_score(fr_t)[0])
   plt.xlim([t0,t1])

   plt.show()
   """

   data_sel=data_sel.loc[data_sel["Time"]<t1]
   data_sel["Vocal"]=0
   data_sel.loc[data_sel["Time"]>t0,"Vocal"]=1
   data_sel=data_sel.loc[~np.isnan(data_sel["Frequency"])]


   harm_lst=[]
   for i in range(nHarmonics):
     harm_lst.append("Harm_"+str(i))
   h = data_sel.loc[:,harm_lst]

   h1=h.as_matrix().reshape(1,h.size)
   magn_min=h1[np.where(h1>-100)].min()

   for i in range(nHarmonics):
      data_sel.loc[data_sel["Harm_"+str(i)]==-100,"Harm_"+str(i)]=magn_min

  
   for i in range(1,nHarmonics):
      data_sel["Harm_"+str(i)]=data_sel["Harm_"+str(i)]-data_sel["Harm_0"]
   data_sel["Harm_0"]=0

   harm_lst.remove("Harm_0")
   data_w=data_sel.loc[:,["Frequency"]+harm_lst+["Vocal"]]
   """
   for i in range(0,data_w.shape[0],1000):
        rec=data_w.iloc[i]
        x = (np.arange(nHarmonics)+1)*rec["Frequency"]
        y = rec[harm_lst].as_matrix()
        if rec["Vocal"]==0:
           col="b"
        else:
           col="r"
           plt.plot(x,y,color=col)
   plt.xlim([0,5000])
   plt.show()
   """

   X=data_w.as_matrix()[:,0:-1]
   scaler=StandardScaler()
#   print X_scaled.mean(axis=0)
#   print X_scaled.std(axis=0)
   y=data_w.as_matrix()[:,-1]
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=241)
   scaler.fit(X_train)
#   X_train = scaler.transform(X_train)
#   X_test = scaler.transform(X_test)
   clf=linear_model.LogisticRegression(C=0.1,random_state=241,penalty='l2')
   clf.fit(X_train,y_train)
#   y_pred=clf.predict_proba(X_test)[:,1]
   y_pred=clf.predict(X_test)
#   y_pred[y_pred<0.5]=0
#   y_pred[y_pred>=0.5]=1
#   np.set_printoptions(threshold='nan')
#   print np.column_stack([y_test,y_pred])
   print "Harm Log-regression",accuracy_score(y_test,y_pred)
#   print clf.coef_
#   print clf.intercept_
#   return clf
   """
   for i in range(0,X_test.shape[0]):
      rec=X_test[i]
      x = (np.arange(nHarmonics)+1)*rec[0]
      y = np.append(np.array([0]),rec[1:])
      if y_test[i]==0:
         col="b"
      else:
         col="r"
      if y_test[i]==0 and y_pred[i]>0.5:   
         print i,y_test[i],y_pred[i]
         if i==2630:
            plt.plot(x,y,color=col)
   plt.xlim([0,5000])
   plt.show()
   """   

#   X_scaled = scaler.transform(X)
   X1=X[:,:]
   kmeans = KMeans(n_clusters=2, random_state=150).fit(X1)

   cl = kmeans.labels_
#   print "center",kmeans.cluster_centers_
   print "Harm cluster",accuracy_score(y,cl)
   """
   for i in range(0,X.shape[0],10):
      if cl[i]==0:
        col='r'
      else:
        col='b'
      plt.scatter(X[i,1],X[i,3],color=col)
   plt.show()
   """
   return
   pca = PCA(n_components=2)
   pca.fit(X1)
   X1_tr=pca.transform(X1)
   print X1_tr
   kmeans = KMeans(n_clusters=2, random_state=151).fit(X1_tr)
   cl = kmeans.labels_
   print accuracy_score(y,cl)
   for i in range(0,X.shape[0],10):
      if y[i]==0:
        col='r'
      else:
        col='b'
      plt.scatter(X1_tr[i,0],X1_tr[i,1],color=col)
   plt.show()

def an_data_t(Filename,clf):
   data=pd.read_table("OP_Se_in_ciel.dat")
   data_sel=data[data.Filename==Filename]
   time=data_sel.loc[:,"Time"].as_matrix()
   freq=data_sel.loc[:,"Frequency"].as_matrix()
   pitch=data_sel.loc[:,"Pitch"].as_matrix()
   pitch[pitch==0]=np.nan
   freq[freq==0]=np.nan
   f0_sm=smooth(freq,win_len,window='hamming')
   data_sel=data_sel.loc[~np.isnan(data_sel["Frequency"])]
   harm_lst=[]
   for i in range(nHarmonics):
     harm_lst.append("Harm_"+str(i))
   h = data_sel.loc[:,harm_lst]

   h1=h.as_matrix().reshape(1,h.size)
   magn_min=h1[np.where(h1>-100)].min()

   for i in range(nHarmonics):
      data_sel.loc[data_sel["Harm_"+str(i)]==-100,"Harm_"+str(i)]=magn_min


   for i in range(1,nHarmonics):
      data_sel["Harm_"+str(i)]=data_sel["Harm_"+str(i)]-data_sel["Harm_0"]


   data_sel["Harm_0"]=0
   harm_lst.remove("Harm_0")
   data_w=data_sel.loc[:,["Frequency"]+harm_lst]
   X=data_w.as_matrix()[:,1:]
   X1=X[:,:]
   kmeans = KMeans(n_clusters=2, random_state=150).fit(X1)
   cl = kmeans.labels_

   y_pred=clf.predict(X)
   print accuracy_score(y_pred,cl)
  
   
   for i in range(0,X.shape[0],30):
      if y_pred[i]==0:
        col='r'
      else:
        col='b'
      plt.scatter(X[i,0],X[i,2],color=col)
   plt.show()

def an_data_p(Filename):
   data=pd.read_table("OP_Se_in_ciel.dat")
   time=data.loc[:,"Time"].as_matrix()
   freq=data.loc[:,"Frequency"].as_matrix()
   pitch=data.loc[:,"Pitch"].as_matrix()
   pitch[pitch==0]=np.nan
   freq[freq==0]=np.nan
   f0_sm=smooth(freq,win_len,window='hamming')
   bark_lst=[]
   bark_lst1=[]
   for i in range(nBarkband):
      bark_lst.append("Barkband_"+str(i))
      bark_lst1.append("Barkband1_"+str(i))


   bark_tbl=data.loc[:,bark_lst].as_matrix()
   sb=np.sum(bark_tbl,axis=1)
   sb[sb==0]=1e-10
   sb=sb.reshape(sb.size,1)
   bark_tbl=bark_tbl/sb/marg_d
   bark_tbl[bark_tbl==0]=10**(-100/10)
   bark_tbl=10*np.log10(bark_tbl)+100

   bark1_tbl=data.loc[:,bark_lst1].as_matrix()
   sb1=np.sum(bark1_tbl,axis=1)
   sb1=sb1.reshape(sb1.size,1)
   bark1_tbl=bark1_tbl/sb1/marg_d
   bark1_tbl[bark1_tbl==0]=10**(-100/10)
   bark1_tbl=10*np.log10(bark1_tbl)+100
   
   t0=49
   t1=72

   X=bark1_tbl[time<t1]
   X_b=bark_tbl[time<t1]
   fr=freq[time<t1]
   time1=time[time<t1]
   y=np.zeros(X.shape[0])
   y[time1>t0]=1
   X=X[~np.isnan(fr)]
   X_b=X_b[~np.isnan(fr)]
   y=y[~np.isnan(fr)]

   X1=X[:,4:22]

   X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.33, random_state=261)
   clf=linear_model.LogisticRegression(C=0.1,random_state=261,penalty='l1')
   clf.fit(X_train,y_train)
#   y_pred=clf.predict_proba(X_test)[:,1]
   y_pred=clf.predict(X_test)
#   y_pred=clf.predict(X_test)
#   y_pred[y_pred<0.5]=0
#   y_pred[y_pred>=0.5]=1
#   np.set_printoptions(threshold='nan')
#   print np.column_stack([y_test,y_pred])
   print "log-regression Barkband1",accuracy_score(y_test,y_pred)
#   print clf.intercept_
#   print np.column_stack([np.arange(4,22),marg[4:22],clf.coef_[0]])

   """ 
   for i in range(0,X.shape[0],1000):
      if y[i]==0:
        col="b"
        plt.plot(marg_m,X[i],color=col) 
      else:
        col="r"
        plt.plot(marg_m,X[i],color=col) 
   plt.xlim([0,5000])
   plt.show()
   """   
   X2=X[:,4:22]
   kmeans = KMeans(n_clusters=2, random_state=100).fit(X2)
   cl = kmeans.labels_
#   print "center",kmeans.cluster_centers_
   print "Clusters BarkBand1",accuracy_score(y,cl)

   pca = PCA(n_components=15)
   pca.fit(X2)
   X2_tr=pca.transform(X2)
   kmeans = KMeans(n_clusters=2, random_state=241).fit(X2_tr)
   cl = kmeans.labels_
#   print "center",kmeans.cluster_centers_
   print "Clusters PCA BarkBand1",accuracy_score(y,cl)

   X3=X_b[:,4:22]
   X_train, X_test, y_train, y_test = train_test_split(X3, y, test_size=0.33, random_state=261)
#   scaler=StandardScaler()
#   scaler.fit(X_train)
#   X_train = scaler.transform(X_train)
#   X_test = scaler.transform(X_test)
   clf=linear_model.LogisticRegression(C=0.1,random_state=261,penalty='l1')
   clf.fit(X_train,y_train)
   y_pred=clf.predict(X_test)
   print "log-regression Barkband",accuracy_score(y_test,y_pred)
#   print clf.intercept_
#   print clf.coef_[0]
#   print np.column_stack([marg[4:22],clf.coef_[0]])
   """
   for i in range(0,X_train.shape[0],100):
      if y_train[i]==0:
        col="b"
        plt.scatter(marg_m[4:22],X_train[i],color=col) 
      else:
        col="r"
        plt.scatter(marg_m[4:22],X_train[i],color=col) 
   plt.xlim([0,5000])
   plt.show()
   """

   kmeans = KMeans(n_clusters=2, random_state=241).fit(X_b)
   cl = kmeans.labels_
#   print "center",kmeans.cluster_centers_
   print "Clusters Barkband",accuracy_score(y,cl)

   mfcc_lst=[]
   for i in range(numberMFCC):
      mfcc_lst.append("MFCC1_"+str(i))
   mfcc_tbl=data.loc[:,mfcc_lst].as_matrix()
   X=mfcc_tbl[time<t1]
   X=X[~np.isnan(fr)]
#   X_mfcc=X[:,1:7]
   X_mfcc=X[:,1:]

   X_train, X_test, y_train, y_test = train_test_split(X_mfcc, y, test_size=0.33, random_state=261)
#  scaler=StandardScaler()
#   scaler.fit(X_train)
#   X_train = scaler.transform(X_train)
#   X_test = scaler.transform(X_test)
   clf=linear_model.LogisticRegression(C=0.1,random_state=261,penalty='l1')
   clf.fit(X_train,y_train)
#   y_pred=clf.predict_proba(X_test)[:,1]
   y_pred=clf.predict(X_test)
#   y_pred=clf.predict(X_test)
#   y_pred[y_pred<0.5]=0
#   y_pred[y_pred>=0.5]=1
#   np.set_printoptions(threshold='nan')
#   print np.column_stack([y_test,y_pred])
   print "log-regression MFCC1",accuracy_score(y_test,y_pred)
   print clf.intercept_
   print clf.coef_[0]
  
   
    
#   scaler=StandardScaler()
#   scaler.fit(X_mfcc)
#   X_mfcc = scaler.transform(X_mfcc)
   kmeans = KMeans(n_clusters=2, random_state=77).fit(X_mfcc)

   cl = kmeans.labels_
#   print "center",kmeans.cluster_centers_
   print "Clusters MFCC1",accuracy_score(y,cl)

   kmeans = KMeans(n_clusters=3, random_state=77).fit(X_mfcc)

   cl = kmeans.labels_
   print cl.min(),cl.max()

   
   for i in range(0,X_mfcc.shape[0],10):
       if cl[i]==0:
          col="b"
       elif cl[i]==1:
          col="r"
       elif cl[i]==2:
          col="g"
       elif cl[i]==3:
          col="cyan"
       plt.scatter(X_mfcc[i,0],X_mfcc[i,1],color=col)
   plt.show()

#   plt.scatter(np.arange(cl.shape[0]),cl)
#   plt.show()


   """
   for i in range(0,X_mfcc.shape[0],10):
       if cl[i]==0:
          col="b"
       else:
          col="r"
       plt.scatter(X_mfcc[i,0],X_mfcc[i,2],color=col)
   plt.show()
   """
   """
   fr=fr[~np.isnan(fr)]
   time2=time1[~np.isnan(fr)]
   plt.plot(time2,cl*300,color='b')
   plt.plot(time2,fr,color='r')
   for i in range(-12,16):
      fr_t=2**(i/12.0)*440
      plt.plot(time2,fr_t*np.ones(time2.size),'black')
      plt.text(0,fr_t,get_score(fr_t)[0])

   plt.show()
   """
def an_data(Filename):
   data=pd.read_table(Filename+".dat")
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
   X=mfcc_tbl
   X1=X[~np.isnan(freq)]
   
   X_mfcc=X1[:,1:]
   kmeans = KMeans(n_clusters=3, random_state=77).fit(X_mfcc)
   cl = kmeans.labels_

   cl_n=np.zeros(freq.shape[0])*np.nan
   j=0
   for i in range(cl_n.shape[0]):
      if not np.isnan(freq[i]):
          cl_n[i]=cl[j]
          j+=1

   harm_lst=[]
   for i in range(nHarmonics):
      harm_lst=np.append(harm_lst,"Harm_"+str(i))

   i_g=0
   for ind in data[cl_n==0].index :
       if i_g%1000==0:
          harm=data.iloc[ind][harm_lst].values
          harm = harm - harm[0]    
          fr0=data.iloc[ind]["Frequency"]
          fr=(np.arange(nHarmonics)+1)*fr0
          plt.plot(fr,harm,color="b")
       i_g+=1

   i_g=0
   for ind in data[cl_n==1].index :
       if i_g%1000==0:
          harm=data.iloc[ind][harm_lst].values
          harm = harm - harm[0]    
          fr0=data.iloc[ind]["Frequency"]
          fr=(np.arange(nHarmonics)+1)*fr0
          plt.plot(fr,harm,color="r")
       i_g+=1

   i_g=0
   for ind in data[cl_n==2].index :
       if i_g%1000==0:
          harm=data.iloc[ind][harm_lst].values
          harm = harm - harm[0]    
          fr0=data.iloc[ind]["Frequency"]
          fr=(np.arange(nHarmonics)+1)*fr0
          plt.plot(fr,harm,color="g")
       i_g+=1

   i_g=0
   for ind in data[cl_n==3].index :
       if i_g%1000==0:
          harm=data.iloc[ind][harm_lst].values
          harm = harm - harm[0]    
          fr0=data.iloc[ind]["Frequency"]
          fr=(np.arange(nHarmonics)+1)*fr0
          plt.plot(fr,harm,color="cyan")
       i_g+=1

   plt.xlim([0,5000])
   plt.show()    

   f0_sm[cl_n==0]=np.nan


#   for seq in  find_seq(f0_sm,0.4):
   cnt=0
   for seq in  find_seq(f0_sm,0.5):
         plt.plot(time[seq[0]:seq[1]],freq[seq[0]:seq[1]],color='b')
         plt.plot(time[seq[0]:seq[1]],f0_sm[seq[0]:seq[1]],color='cyan')
         plt.plot([time[seq[0]],time[seq[0]]],[0,1000],color="g")
         plt.plot([time[seq[1]],time[seq[1]]],[0,1000],color="r")
         t0=time[seq[0]]
         t1=time[seq[1]-1]
         syn_sound(data.loc[seq[0]:seq[1]-1,:],"sounds/"+"%02d" % cnt+"_"+Filename+".wav")
#         cmd="ffmpeg -y -i "+Filename+".flac -ss "+"%.2f" % t0 +" -t "+"%.2f" % (t1-t0)  +" "+"%02d" % cnt+"_"+Filename+".wav"
#         print cmd
#         os.system(cmd)
         cnt+=1
   print cnt
   for i in range(-12,16):
      fr_t=2**(i/12.0)*440
      plt.plot(time,fr_t*np.ones(time.size),'black')
      plt.text(0,fr_t,get_score(fr_t)[0])
#   plt.xlim((50,100))
   plt.show()

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

   data=pd.read_table("OP_Se_in_ciel.dat")

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
   data=pd.read_table(Filename+".dat")
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

   mfcc_tbl_sing=np.empty((0,numberMFCC))
   f0_sm_sing=[]

   f0_sm_m_l=[]
   harm_m_l=np.empty((0,nHarmonics))
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
# middle spectrum
#         plt.plot((np.arange(nHarmonics)+1)*f0_sm_m,harm_m,color=col)

#         plt.plot([time[seq[0]],time[seq[0]-t0]],[0,1000],color="g")
#         plt.plot([time[seq[1]],time[seq[1]]],[0,1000],color="r")
# the syntesis of sound
#         syn_sound(data.loc[seq[0]:seq[1]-1,:],"sounds/"+"%02d" % cnt+"_"+Filename+".wav")

         mfcc_tbl_sing=np.vstack([mfcc_tbl_sing,mfcc_tbl_sm[seq[0]:seq[1],:]]) 
         f0_sm_sing=np.append(f0_sm_sing,f0_sm[seq[0]:seq[1]])
         cnt+=1

#   plt.xlim((0,5000))
#   plt.show()
#   plt.hist(f0_sm_m_l)
#   plt.show()

   regr = linear_model.LinearRegression()
   X_train, X_test, y_train, y_test = train_test_split(mfcc_tbl_sing[:,1:10], f0_sm_sing, test_size=0.33, random_state=241)
   regr.fit(X_train,y_train)

#   print "Error=",np.sqrt(np.mean((regr.predict(X_test) - y_test) ** 2))
#   plt.scatter(regr.predict(X_test),y_test)
#   plt.show()
   return f0_sm_m_l,harm_m_l

nHarmonics=15
frameSize = 2*1024
hopSize = 256 
sampleRate = 44100
nBarkband=27
numberMFCC=16
marg=[0,50,100,150,200,300,400,510,630,770,920,1080,1270,1480,1720,2000,2320,2700,3150,3700,4400,5300,6400,7700,9500,12000,15500,20500]#,27000]

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


harm_m=np.zeros((nBarkband,nHarmonics))
cnt_m=np.zeros(nBarkband)
for tr in tr_lst[0:]:
    f0,harm = an_data_use(tr,clf)
#    print np.min(f0),np.max(f0)
    for i in range(len(f0)): 
        for i_m in range(len(marg)):
           if marg[i_m]>f0[i]:
              break
   

        harm_s=harm_m[i_m-1]
        harm_s+=harm[i]
        harm_m[i_m-1]=harm_s

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
            print i_m,clr,marg_m[i_m],cnt_m[i_m],get_score(marg_m[i_m])[0]
            harm_m[i_m]=harm_m[i_m]/cnt_m[i_m]
            plt.plot((np.arange(nHarmonics)+1)*marg_m[i_m],harm_m[i_m],color=clr)

plt.xlim((0,5000))
plt.show()
