import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import essentia as es
from essentia.standard import *


def find_seq(seq):
   k_t=hopSize/float(sampleRate)
   isPitch=False
   delta_t=int(0.4/k_t)
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
           i_end=i+1
           if i_end-i_start>delta_t:
              acc_pitch.append([i_start,i_end,pitch])
           isPitch=False
         else:
           if get_score(f)!=pitch:
              i_end=i+1
              if i_end-i_start>delta_t:
                 acc_pitch.append([i_start,i_end,pitch])
              f_start=f
              pitch=get_score(f_start)
              i_start=i
 
   if isPitch:
      i_end=i+1
      if i_end-i_start>delta_t:
         acc_pitch.append([i_start,i_end])
         acc_pitch.append([i_start,i_end,pitch])
   return acc_pitch


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


def an_data(Filename):
   data=pd.read_table("Peretyatko.dat")
   data_sel=data[data.Filename==Filename]
   t=data_sel.loc[:,"Time"].as_matrix()
   freq=data_sel.loc[:,"Frequency"].as_matrix()
   pitch=data_sel.loc[:,"Pitch"].as_matrix()
   freq[freq==0]=np.nan
   f0_sm=smooth(freq,win_len,window='hamming')

   centroid=data_sel.loc[:,"Centroid"].as_matrix()
   centroid[np.isnan(freq)]=np.nan


   h_lst=[]
   for i in range(nHarmonics):   
      h_lst.append("Harm_"+str(i))
   harm=data_sel.loc[:,h_lst].as_matrix()


   harm_min=np.min(harm[harm>-100])

   harm[harm==-100]=-74#harm_min
   harm_a=10**(harm/20)
   harm_sm=np.empty((0,harm.shape[0]))
   for i in range(nHarmonics): 
      hm=smooth(harm_a[:,i],win_len,window='hamming')
      harm_sm=np.append(harm_sm,[20*np.log10(hm)],axis=0)
   harm_sm=harm_sm.transpose()
   """
   for seq in  find_seq(f0_sm):
       if seq[2][0]=='a-1':
          plt.plot(t[seq[0]:seq[1]],freq[seq[0]:seq[1]])
          plt.plot(t[seq[0]:seq[1]],f0_sm[seq[0]:seq[1]])
   for i in range(-12,16):
      fr_t=2**(i/12.0)*440
      plt.plot(t,fr_t*np.ones(t.size),'black')
      plt.text(0,fr_t,get_score(fr_t)[0])

#   plt.xlim([1.5,t[-1]])
   plt.show()
   """
#   cnt=1
   global cnt
   print cnt
   for seq in  find_seq(f0_sm):
#     for pt in ['c-1','c#-1','d-1','d#-1','e-1','f-1','f#-1','g-1','g#-1','a-1','a#-1','h-1','c-2','c#-2','d-2','d#-2','e-2','f-2','f#-2','g-2','g#-2','a-2','a#-2']:
     for pt in ['f-2']:
       if seq[2][0] == pt :
           pt=seq[2][0]
           h=harm_sm[seq[0]:seq[1],:]
           for k in range(nHarmonics):
               h[:,k]=h[:,k]-h[:,0]
#           for i in range(h.shape[0]):
#             plt.plot(np.arange(nHarmonics)+1,h[i])
           if cnt==1:
              color='red'
           elif cnt==2:
              color='orange'
           elif cnt==3:
              color='yellow'
           elif cnt==4:
              color='g'
           elif cnt==5:
              color='lightblue'
           elif cnt==6:
              color='b'
           elif cnt==7:
              color='violet'
           else:
              color='b'             
           plt.plot(np.arange(nHarmonics)+1,np.mean(h,axis=0),color=color)
           print pt,Filename,cnt
           """ 
           loader = MonoLoader(filename = Filename[0:-4]+"_harm.wav", sampleRate = sampleRate )
           audioWriter = MonoWriter(filename=str(cnt).zfill(2)+"_"+Filename[0:-4]+"_"+pt+"_p.wav")
           slicer=Slicer(startTimes=[seq[0]*k_t],endTimes=[seq[1]*k_t],sampleRate=sampleRate)
           audio=slicer(loader())[0]
           audioWriter(audio.astype(np.float32))
           """
           cnt=cnt+1
#       plt.show()    


           
      
#   for seq in  find_seq(f0_sm):
#      plt.plot(t[seq[0]:seq[1]],harm_sm[seq[0]:seq[1],0])
#   plt.show()

margBark=[0,50,100,150,200,300,400,510,630,770,920,1080,1270,1480,1720,2000,2320, 2700,3150,3700,4400,5300,6400,7700,9500,12000,15500,20500]#,27000]
nBarkBand=len(margBark)-1
marg_d=[]
marg_m=[]
for i in range(nBarkBand):
   marg_m=np.append(marg_m,(margBark[i]+margBark[i+1])/2.0)
   marg_d=np.append(marg_d,margBark[i+1]-margBark[i])

nHarmonics=15
frameSize = 2*1024
hopSize = 256 
sampleRate = 44100

k_t=hopSize/float(sampleRate)
win_len=99
global cnt
cnt=1
an_data("OP_SeInCiela.wav")
an_data("OP_NonMiDira.wav")
an_data("OP_Bacio1a.wav")
an_data("OP_Bacio2a.wav")
an_data("OP_Bacio3a.wav")

plt.show()    


