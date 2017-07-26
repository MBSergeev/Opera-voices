import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import essentia as es
from essentia.standard import *

def singer_color(sng):
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
     else:
       col="magenta"
     return col



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

def get_freq(pitch):
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

    for fr in zip(scores,freq):
        if fr[0]==pitch:
           return fr[1]

    return None

data=pd.read_table("Adina_bark.dat")


def work(data,sng):
   print sng
   data=data[data.loc[:,"Singer"]==sng]
   t_arr=data.loc[:,"Time"].as_matrix()
   f0_sm=data.loc[:,"Mean_frequency"].as_matrix()
   f=data.loc[:,"Frequency"].as_matrix()
   pitch=data.loc[:,"Pitch"].as_matrix()



#   plt.plot(t_arr,f,"b")
#   plt.plot(t_arr,f0_sm,"r")
#   plt.plot(t_arr,pitch,"g")
#   for i in range(-12,16):
#       fr_t=2**(i/12.0)*440
#       plt.plot(t_arr,fr_t*np.ones(len(t_arr)),'black')
#       plt.text(0,fr_t,get_score(fr_t)[0])

#   plt.show()


#plt.plot(t_arr,f0_sm/10,"black")
   for i in range(1,nHarm):
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
         mng=data.loc[:,"Harm_"+str(i)].as_matrix()
#         plt.plot(t_arr,mng,color=clr)


   #plt.show()

   nBarkband=27
   marg=[0,50,100,150,200,300,400,510,630,770,920,1080,1270,1480,1720,2000,2320,2700,3150,3700,4400,5300,6400,7700,9500,12000,15500,20500]#,27000]
   lst=[]
   for i in range(nBarkband):
      lst.append("BarkBand_"+str(i))
   lst1=[]
   for i in range(nBarkband):
      lst1.append("BarkBand1_"+str(i))
   barkband=data[lst]
   barkband1=data[lst1]
   fr_a=[]
   fr_d=[]
   for i in range(len(marg)-1):
      fr_a=np.append(fr_a,(marg[i]+marg[i+1])/2.0)
      fr_d=np.append(fr_d,marg[i+1]-marg[i])
   nFrames=barkband.shape[0]
   f_lst=[200,400,600,800,2000]
   f_lst=[200,300,400,510,630,220,920,1080,2000]
   f_lst=[300,1000]
#   koef=2**(1.0/24)
   koef=2**(1.0/12)
#   fr_c=get_freq("d#-2")
#   f_lst=[fr_c/koef,fr_c*koef]
   f_lst=[]
#   for fr_t in ["e-1","a-1","e-2","g-2","h-2"]:
   for fr_t in ["a#-1","d#-2","a#-2"]:
      fr_c=get_freq(fr_t)
      f_lst.append(fr_c/koef)
      f_lst.append(fr_c*koef)

   pkd=PeakDetection()
   lst_h=[]
   for i in range(1,nHarm):
      lst_h.append("Harm_"+str(i))
   harm = data.loc[:,lst_h].as_matrix()

   for n_fr in range(nFrames):
      if np.isnan(f0_sm[n_fr]):
         continue
      pos,amp=pkd(es.array(harm[n_fr]))
      if pos.shape[0]==1 or (round(pos[0],2)==0 and round(pos[1],2)==1):
         f0_sm[n_fr]=np.nan    
   X=np.empty((0,nHarm-1))
   f0_sm1=[]
   for i_f in range(len(f_lst)-1):# True:
     s_bark=np.zeros(barkband.shape[1])
     s_bark1=np.zeros(barkband1.shape[1])
     cnt=0
     for n_fr in range(nFrames):
         if np.isnan(f0_sm[n_fr]):
           continue
         if not (f_lst[i_f]<=f0_sm[n_fr]<f_lst[i_f+1]):
           continue 

         br=barkband.iloc[n_fr].as_matrix()
         sm=np.sum(br)
         br=br/sm
         s_bark+=br
         br1=barkband1.iloc[n_fr].as_matrix()
         sm1=np.sum(br1)
         br1=br1/sm1 
         s_bark1+=br1
         fr_h=f0_sm[n_fr]*np.arange(1,nHarm+1) 
         col=singer_color(sng)
#         plt.plot(fr_h,np.append(np.array(0),harm[n_fr]),color=col)
         X=np.append(X,[harm[n_fr]],axis=0)
         f0_sm1=np.append(f0_sm1,f0_sm[n_fr])
         cnt+=1
         
         

#      br[br==0]=10**(-80/10)
#      br1[br1==0]=10**(-80/10)
#      br=10*np.log10(br)+80
#      br1=10*np.log10(br1)+80
#      if n_fr==300:
#        plt.plot(fr_a,br)
#        plt.plot(fr_a,br1)
#        plt.bar(marg[0:-1],br,width=fr_d,color='b',alpha=0.5)
#        plt.bar(marg[0:-1],br1,width=fr_d,color='g',alpha=0.5)
#        for i in range(16):
#           f=f0_sm[n_fr]*(i+1)
#           plt.plot([f,f],[0,-10])
#        plt.xlim([0,6000])
#        plt.show()

     s_bark/=cnt
     sm=np.sum(s_bark)
     if sm>0:
        s_bark=s_bark/sm/fr_d

     s_bark1/=cnt
     sm1=np.sum(s_bark1)
     if sm1>0:
        s_bark1=s_bark1/sm1/fr_d

     s_bark[s_bark==0]=10**(-90/10)
     s_bark1[s_bark1==0]=10**(-90/10)
     s_bark=10*np.log10(s_bark)
     s_bark1=10*np.log10(s_bark1)
     col=singer_color(sng)

#     plt.plot(fr_a,s_bark,color=col)
#     plt.plot(fr_a,s_bark1,color=col)
     """
     from sklearn.cluster import KMeans
     n_clusters=2
     kmeans=KMeans(n_clusters=n_clusters,random_state=42)
     kmeans.fit(X)
     pred=kmeans.predict(X)
     fr_h=fr_c*np.arange(1,nHarm+1) 
     for i in range(X.shape[0]):
        if pred[i]==0:
            col='b'
        elif pred[i]==1:
            col='r'
        elif pred[i]==2:
            col='g'
        elif pred[i]==3:
            col='cyan'
        plt.plot(fr_h,np.append(np.array(0),X[i]),color=col)
      """      
     X_mean0=np.mean(X,axis=0)
     X_mean1=np.mean(X,axis=1)
     X_mean=np.mean(X)
     s_delta=0
     for i in range(X.shape[0]):
        fr_h=f0_sm1[i]*np.arange(1,nHarm+1) 
        harm_pred=[]
        for j in range(X.shape[1]):
           X_pred=X_mean0[j]+X_mean1[i]-X_mean
           harm_pred=np.append(harm_pred,X_pred)
           delta = X[i,j]-X_pred 
           s_delta+=delta*delta
#        plt.plot(fr_h,np.append(np.array(0),harm_pred),'b')   
#        plt.plot(fr_h,np.append(np.array(0),X[i]),'r')   
     s_delta=np.sqrt(s_delta/(X.shape[0]*X.shape[1]-X.shape[0]-X.shape[1]-1))
     print "sigma=",s_delta
#     plt.plot(fr_h,np.append(np.array(0),X_mean0),col)
     if i_f%2==0:
        plt.plot(fr_h,np.append(np.array(0),X_mean0)-X_mean0[0],col)
     
        

nHarm=15
data=pd.read_table("Adina_bark.dat")
#data=pd.read_table("Bacio_bark.dat")
sng_lst=["AN","OP","VN","LP","MD","JS","AG","AK","LC","AF","IC","KB","ES","RS"]
sng_lst=["OP","VN"]
for sng in sng_lst:
   work(data,sng)

plt.xlim([0,6000])
plt.show()

# AN - Anna Netrebko
# OP - Olga Peretyatko
# VN - Valentina Nafornita 
# LP - Lucia Popp
# MD - Mariella Devia
# JS - Joan Sutherland
# AG - Angela Gheorghiu
# AK - Alexandra Kurzak
# LC - Lucy Crowe
# AF - Alida Ferrarini
# IC - Ileana Cotrubas
# KB - Kathleen Battle
# ES - Ekaterina Siurina
# RS - Renata Scotto
