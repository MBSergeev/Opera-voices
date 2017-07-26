import essentia as es
import os
import os.path
import fnmatch
from essentia.standard import *

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
    return min_f[0]

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


def anal_pitch(path,inputFilename):
   frameSize = 2048
   hopSize = 256 
   sampleRate = 44100
   minF0=200.0
   maxF0=2000.0
   minSineDur = 0.02
   harmDevSlope=0.01
   win_len=51

# functions
   loader = MonoLoader(filename = path+"/"+inputFilename, sampleRate = sampleRate )
   window =Windowing(type='hamming')
   fft=FFT(size=frameSize)
   spectrum=Spectrum(size=frameSize)
   equalLoudness = EqualLoudness()
   predominantMelody=PredominantPitchMelodia(frameSize=frameSize,hopSize=hopSize,sampleRate=sampleRate)
   sct=SpectralCentroidTime(sampleRate=sampleRate)
   mfcc=MFCC(numberCoefficients=numberMFCC,inputSize=frameSize/2+1)
   harmonicmodelanal=HarmonicModelAnal(sampleRate=sampleRate,hopSize=hopSize,nHarmonics=nHarmonics,harmDevSlope=harmDevSlope,
       maxFrequency=maxF0,minFrequency=minF0)


   audio=loader()
#equal loudness
   eqaudio=equalLoudness(audio)
# pitch
   pitch,pitchConfidence=predominantMelody(eqaudio)
   for p in enumerate(pitch):
      if p[1]==0:
         pitch[p[0]]=np.nan
      if p[1]>1000:
         pitch[p[0]]=np.nan
       
   

   allMfcc=np.empty((0,numberMFCC))
   allMfcc1=np.empty((0,numberMFCC))
   allCentroids=[]
   cnt=0
# The cycle 
   for frame in FrameGenerator(audio, frameSize, hopSize, startFromZero=False):
# SpectralCentroid
       centroid=sct(frame)
# Spectrum
       spect=spectrum(window(frame))
# MFCC-coeffitients
       mfcc_bands,mfcc_coeffs=mfcc(spect)
# Fourie transformation
       infft = fft(window(frame))
# harmonic analysis

       frequencies,magnitudes,phases=harmonicmodelanal(infft,pitch[cnt])

       frequencies[0]=pitch[cnt]
       if frequencies[0]==0:
          frequencies[0]=np.nan
          centroid=np.nan

       for i in range(len(frequencies)):
          frequencies[i]=frequencies[0]*(i+1)
       
       spect1=np.zeros(spect.shape)
       for i in range(len(magnitudes)):
          if not np.isnan(frequencies[i]) and not np.isnan(magnitudes[i]):
#             print cnt,i,pitch[cnt],frequencies[i],frequencies[i]*frameSize/sampleRate,magnitudes[i]
             spect1[int(frequencies[i]*frameSize/sampleRate)]=10**(magnitudes[i]/20)
       

       mfcc_bands1,mfcc_coeffs1=mfcc(es.array(spect1))

# Sum
       allCentroids=np.append(allCentroids,centroid)
       allMfcc=np.append(allMfcc,[mfcc_coeffs],axis=0)
       allMfcc1=np.append(allMfcc1,[mfcc_coeffs1],axis=0)
       cnt+=1

   nFrames=cnt

   t_arr=np.arange(nFrames)*hopSize/float(sampleRate)

# Smooth of basic frequency
   f_mean=np.mean(pitch[~np.isnan(pitch)])
   f0_sm=f_mean*np.ones(nFrames)

# plot basic frequency
#   plt.plot(t_arr,pitch,'b')
#   plt.plot(t_arr,allCentroids,'g')
#   plt.plot(t_arr,f0_sm,'r')
#   plt.show()

# plot MFCC
#   for i in range(0,numberMFCC):
#      print "mfcc=",i
#      plt.plot(t_arr,allMfcc[:,i],'b')
#      plt.plot(t_arr,allMfcc1[:,i],'r')
#      plt.show()

# smooth MFCC
   mf=smooth(allMfcc[:,0],win_len,window='hamming')
   mf1=smooth(allMfcc[:,0],win_len,window='hamming')
   m_mf=np.empty((0,len(mf)))
   m_mf1=np.empty((0,len(mf)))
   mf_mid=[]
   mf_mid1=[]
   for i in range(numberMFCC):
      mf=smooth(allMfcc[:,i],win_len,window='hamming')
      m_mf=np.append(m_mf,[mf],axis=0)
      mf_mid=np.append(mf_mid,np.mean(m_mf[i]))
      mf1=smooth(allMfcc1[:,i],win_len,window='hamming')
      m_mf1=np.append(m_mf1,[mf1],axis=0)
      mf_mid1=np.append(mf_mid1,np.mean(m_mf1[i]))

#   for i in range(0,numberMFCC):
#      print "mfcc=",i
#      plt.plot(t_arr,m_mf[i],'b')
#      plt.plot(t_arr,mf_mid[i]*np.ones(nFrames),'b')

#      plt.plot(t_arr,m_mf1[i],'r')
#      plt.show()

   ptch=inputFilename[-7:-4]
   if ptch[0]=='#':
      ptch=inputFilename[-8:-4]
   pitch_fr=get_freq(ptch)
   if pitch_fr/f_mean>1.2 or f_mean/pitch_fr>1.2:
       return   

   fp=open(outputFile,"a")
   fp.write(inputFilename)

   fp.write("\t"+inputFilename[0:2])
   fp.write("\t"+str(pitch_fr))
   fp.write("\t"+str(f_mean))
   for i in range(1,numberMFCC):
      fp.write("\t"+str(mf_mid[i]))
      fp.write("\t"+str(mf_mid1[i]))

   fp.write("\n")
   fp.close()

   return

         
nHarmonics=15
numberMFCC=16

outputFile="Adina_mfcc.dat"
if os.path.exists(outputFile):
    os.remove(outputFile)
   
    
fp=open(outputFile,"w")
fp.write("Filename\tSinger\tPitch_frequency\tMean_frequency")
for i in range(1,numberMFCC):
   fp.write("\tMFCC_"+str(i+1))
   fp.write("\tMFCC1_"+str(i+1))
fp.write("\n")
fp.close()


pt='sounds'

lst=fnmatch.filter(os.listdir(pt), 'OP_Adina_000_a-1.wav')
lst=fnmatch.filter(os.listdir(pt), 'OP_Adina_*.wav')
lst=fnmatch.filter(os.listdir(pt), '??_Adina_*.wav')
#lst=fnmatch.filter(os.listdir(pt), 'LC_Adina_011_c-2.wav')
#lst=fnmatch.filter(os.listdir(pt), 'LC_Adina_*.wav')
lst.sort()
for f in lst:
   print f
   anal_pitch(pt,f)
