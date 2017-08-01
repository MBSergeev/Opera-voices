import numpy as np
import essentia as es
from essentia.standard import *

hopSize = 256
sampleRate = 44100
frameSize = 2*1024
nHarmonics=15

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


def syn_sound(data,name):
   sinemodelsynth = SineModelSynth(sampleRate=sampleRate,fftSize=frameSize,hopSize=hopSize)
   ifft=IFFT(size=frameSize)
   overlapAdd = OverlapAdd(frameSize=frameSize, hopSize=hopSize,  gain=1./frameSize)
   audioWriter = MonoWriter(filename=name)
   magn_lst=[]
   phase_lst=[]

   for i in range(nHarmonics):
      magn_lst=np.append(magn_lst,"Harm_"+str(i))
      phase_lst=np.append(magn_lst,"Phase_"+str(i))

   pool=es.Pool()


   for cnt in range(data.shape[0]):
       if cnt>0 and cnt%10000==0:
          print cnt,data.shape[0]
       dt=data.iloc[cnt,:]
       magn=dt[magn_lst]
       phase=dt[phase_lst]
       freq=[]
       fr0=dt['Frequency']
       freq=fr0*(np.arange(nHarmonics)+1)
       sfftframe=sinemodelsynth(es.array(magn),es.array(freq),es.array(phase))
       ifftframe=ifft(sfftframe)
       out=overlapAdd(ifftframe)
       pool.add('frames',out)

   audioout = pool['frames'].flatten()

# Write to file
   audioWriter(audioout.astype(np.float32))

