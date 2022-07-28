import essentia as es
from essentia.standard import *

import numpy as np
import matplotlib.pyplot as plt
import sys
import fnmatch
import os
from utils import get_score



from my_cons import my_cons


def work(inputFilename,col='b'):
   loader = MonoLoader(filename = my_c.dir_in_sound+inputFilename, sampleRate = my_c.sampleRate )

   equalLoudness = EqualLoudness()
   predominantMelody=PredominantPitchMelodia(frameSize=my_c.frameSize,hopSize=my_c.hopSize,sampleRate=my_c.sampleRate)
   window =Windowing(type='hamming')
   spectrum=Spectrum(size=my_c.frameSize)
   mmf=MaxMagFreq(sampleRate=my_c.sampleRate)
   sp=SpectralPeaks(sampleRate=my_c.sampleRate,orderBy="frequency",magnitudeThreshold=0.01)
   hma=HprModelAnal(sampleRate=my_c.sampleRate,hopSize=my_c.hopSize,fftSize=my_c.frameSize,nHarmonics=my_c.nHarmonics)

   k_t=my_c.hopSize/float(my_c.sampleRate)

   audio=loader()
   eqaudio=equalLoudness(audio)
   pitch,pitchConfidence=predominantMelody(eqaudio)
   cnt=0
   allFrequencies=np.empty((0,my_c.nHarmonics))
   allMagnitudes=np.empty((0,my_c.nHarmonics))
   allPhases=np.empty((0,my_c.nHarmonics))

   freq=[]

   pitch[pitch==0]=np.nan
   pos1=inputFilename.find('_')
   pos2=inputFilename.find('_',pos1+1)
   voice=inputFilename[0:pos1]

   for frame in FrameGenerator(audio, my_c.frameSize, my_c.hopSize, startFromZero=False):
      spect=spectrum(window(frame))
      
      fr,mgn=sp(spect) 
      #print (fr[0],pitch[cnt])  
      freq=np.append(freq,fr[0])
      
      frequencies,magnitudes,phases,res=hma(frame,fr[0])

      frequencies[frequencies==0]=np.nan
      magnitudes[frequencies==0]=np.nan
      phases[frequencies==0]=np.nan
      allFrequencies=np.append(allFrequencies,[frequencies],axis=0)
      allMagnitudes=np.append(allMagnitudes,[magnitudes],axis=0)
      allPhases=np.append(allPhases,[phases],axis=0)
      cnt+=1

   #plt.plot(k_t*(np.arange(cnt)-1),allMagnitudes[:,0])
   #plt.show()
   f_mean=np.mean(pitch[~np.isnan(pitch)])
   fr_mean=np.mean(freq[~np.isnan(freq)])
   fr0_mean=np.mean(allFrequencies[~np.isnan(allFrequencies[:,0]),0])
   n=int(inputFilename[-5])
   snd=inputFilename[-6]
   grp=inputFilename[pos1+1:pos2]
   if grp=='low':
      col='r'
   elif grp=='mid':
      col='g'
      n=n+5
   elif grp=='high':
      col='b'
      n=n+10
   
   sys.stdout.write(voice+"\t"+snd+"\t"+str(n)+"\t"+grp+"\t"+\
      str(f_mean)+"\t"+str(fr_mean)+"\t"+str(fr0_mean))
   for i in range(my_c.nHarmonics):
       sys.stdout.write("\t"+str(np.mean(allMagnitudes[~np.isnan(allMagnitudes[:,i]),i])))  
   sys.stdout.write("\n") 
#   plt.scatter(n,fr_mean,color='b')
#   plt.scatter(n,f_mean,color='r')

#   plt.plot(k_t*(np.arange(cnt)-1),allFrequencies[:,0],color='r')
#   plt.plot(k_t*(np.arange(cnt)-1),freq,color='g')

my_c= my_cons()

#lst=fnmatch.filter(os.listdir(my_c.dir_in_sound), 'Soprano*.wav')
#lst=fnmatch.filter(os.listdir(my_c.dir_in_sound), 'Mezzo*.wav')
#lst=fnmatch.filter(os.listdir(my_c.dir_in_sound), 'Tenor*.wav')
#lst=fnmatch.filter(os.listdir(my_c.dir_in_sound), 'Baritone*.wav')
lst=fnmatch.filter(os.listdir(my_c.dir_in_sound), '*.wav')
lst.sort()
sys.stdout.write( "voice\tsound\tn\tgrp\tf_mean\tfr_mean\tfr0_mean")
for i in range(my_c.nHarmonics):
   sys.stdout.write("\tnHarm_"+str(i+1))
sys.stdout.write("\n")
for f in lst:   
   work(f)
   
    
#plt.show()
