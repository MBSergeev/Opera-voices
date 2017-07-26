import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


def an_data():
   data=pd.read_table("Voice.dat")
#   data_sel=data
   data_sel=data[data.Filename=='OP_strano.wav']
#   data_sel=data[data.Singer=='MD']
   t=data_sel.loc[:,"Time"].as_matrix()
   freq=data_sel.loc[:,"Frequency"].as_matrix()
   pitch=data_sel.loc[:,"Pitch"].as_matrix()
   freq[freq==0]=np.nan
   f0_sm=smooth(freq,99,window='hamming')

   centroid=data_sel.loc[:,"Centroid"].as_matrix()
   centroid[np.isnan(freq)]=np.nan


   h_lst=[]
   for i in range(nHarmonics):   
      h_lst.append("Harm_"+str(i))
   harm=data_sel.loc[:,h_lst].as_matrix()
   harm[harm==-100]=np.nan
   

   mfcc_lst=[]
   for i in range(numberMFCC):
      mfcc_lst.append("MFCC_"+str(i))
   
   mfcc=data_sel.loc[:,mfcc_lst].as_matrix()

   mfcc1_lst=[]
   for i in range(numberMFCC):
      mfcc1_lst.append("MFCC1_"+str(i))
   
   mfcc1=data_sel.loc[:,mfcc1_lst].as_matrix()

   bark_lst=[]
   bark1_lst=[]
   for i in range(nBarkBand):
      bark_lst.append("BarkBand_"+str(i))
      bark1_lst.append("BarkBand1_"+str(i))

   bark=data_sel.loc[:,bark_lst].as_matrix()
   bark_s=bark[1000] 
   sb=np.sum(bark_s)
   if sb>0:
       bark_s=bark_s/sb/marg_d
   bark_s=10*np.log10(bark_s)+100

   bark1=data_sel.loc[:,bark1_lst].as_matrix()
   bark1_s=bark1[1000] 
   sb1=np.sum(bark1_s)
   if sb1>0:
      bark1_s=bark1_s/sb1/marg_d
   bark1_s[bark1_s==0]=10**(-100/10)
   bark1_s=10*np.log10(bark1_s)+100


   plt.plot(t,freq)
   plt.plot(t,f0_sm)
#   plt.plot(t,centroid)
#   plt.plot(t,pitch)
   plt.show()

   plt.plot(t,harm[:,0])
   plt.plot(t,harm[:,1])
   plt.show()

   plt.plot(t,mfcc[:,1])
   plt.plot(t,mfcc1[:,1])
   plt.show()

#   plt.plot(marg_m,bark_s)
   for i in range(len(bark_s)):
      print bark_s[i],bark1_s[i]
   for i in range(nBarkBand):
      plt.bar(margBark[i],bark1_s[i],margBark[i+1]-margBark[i],color='r')
      plt.bar(margBark[i],bark_s[i],margBark[i+1]-margBark[i],color='b',alpha=0.4)
   plt.xlim([0,6000])
   plt.show()

margBark=[0,50,100,150,200,300,400,510,630,770,920,1080,1270,1480,1720,2000,2320, 2700,3150,3700,4400,5300,6400,7700,9500,12000,15500,20500]#,27000]
nBarkBand=len(margBark)-1
marg_d=[]
marg_m=[]
for i in range(nBarkBand):
   marg_m=np.append(marg_m,(margBark[i]+margBark[i+1])/2.0)
   marg_d=np.append(marg_d,margBark[i+1]-margBark[i])
nHarmonics=15
numberMFCC=12
an_data()



