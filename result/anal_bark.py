import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import essentia as es
from essentia.standard import *


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
   for i in range(1,15):
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
   fr_c=get_freq("a#-1")
   koef=2**(1.0/24)
   f_lst=[fr_c/koef,fr_c*koef]
   f_lst=[300,1000]

   pkd=PeakDetection()
   lst_h=[]
   for i in range(1,15):
      lst_h.append("Harm_"+str(i))
   X = data.loc[:,lst_h].as_matrix()

   for n_fr in range(nFrames):
      if np.isnan(f0_sm[n_fr]):
         continue
      pos,amp=pkd(es.array(X[n_fr]))
      if pos.shape[0]==1 or (round(pos[0],2)==0 and round(pos[1],2)==1):
         f0_sm[n_fr]=np.nan    

   for i in range(len(f_lst)-1):# True:
     s_bark=np.zeros(barkband.shape[1])
     s_bark1=np.zeros(barkband1.shape[1])
     cnt=0
     for n_fr in range(nFrames):
         if np.isnan(f0_sm[n_fr]):
           continue
         if not (f_lst[i]<=f0_sm[n_fr]<f_lst[i+1]):
           continue 

         br=barkband.iloc[n_fr].as_matrix()
         sm=np.sum(br)
         br=br/sm
         s_bark+=br
         br1=barkband1.iloc[n_fr].as_matrix()
         sm1=np.sum(br1)
         br1=br1/sm1 
         s_bark1+=br1
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
     if i==0:
         col="red"
     elif i==1:
         col="orange"
     elif i==2:
         col="yellow"
     elif i==3:
         col="green"
     elif i==4:
         col="lightblue"
     elif i==5:
         col="blue"
     elif i==6:
         col="magenta"
     elif i==6:
         col="black"
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

#     plt.plot(fr_a,s_bark,color=col)
     plt.plot(fr_a,s_bark1,color=col)



data=pd.read_table("Adina_bark.dat")
sng_lst=["AN","OP","VN","LP","MD","JS","AG","AK","LC","AF","IC","KB","ES","RS"]
#sng_lst=["LP","AN","MD"]
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
