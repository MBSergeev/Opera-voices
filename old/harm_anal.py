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
   outputFilename=inputFilename[0:len(inputFilename)-4]+"_harm.wav"
   frameSize = 2048
   hopSize = 256 
   sampleRate = 44100
   minF0=200.0
   maxF0=2000.0
   minSineDur = 0.02
#   nHarmonics=15
   harmDevSlope=0.01
   numberMFCC=12
   win_len=51

# functions
   loader = MonoLoader(filename = path+"/"+inputFilename, sampleRate = sampleRate )
   window =Windowing(type='hamming')
   fft=FFT(size=frameSize)
   spectrum=Spectrum(size=frameSize)
   equalLoudness = EqualLoudness()
   predominantMelody=PredominantPitchMelodia(frameSize=frameSize,hopSize=hopSize,sampleRate=sampleRate)
   harmonicmodelanal=HarmonicModelAnal(sampleRate=sampleRate,hopSize=hopSize,nHarmonics=nHarmonics,harmDevSlope=harmDevSlope,
    maxFrequency=maxF0,minFrequency=minF0)

   sinemodelsynth = SineModelSynth(sampleRate=sampleRate,fftSize=frameSize,hopSize=hopSize)
   ifft=IFFT(size=frameSize)
   overlapAdd = OverlapAdd(frameSize=frameSize, hopSize=hopSize,  gain=1./frameSize)
   audioWriter = MonoWriter(filename=path+"/"+outputFilename)
   sct=SpectralCentroidTime(sampleRate=sampleRate)
   mfcc=MFCC(numberCoefficients=numberMFCC,inputSize=frameSize/2+1)

   audio=loader()
#equal loudness
   eqaudio=equalLoudness(audio)
# pitch
   pitch,pitchConfidence=predominantMelody(eqaudio)
   for p in enumerate(pitch):
      if p[1]==0:
         pitch[p[0]]=np.nan

   allFrequencies=np.empty((0,nHarmonics))
   allMagnitudes=np.empty((0,nHarmonics))
   allPhases=np.empty((0,nHarmonics))
   allCentroids=[]
   allMfcc=np.empty((0,numberMFCC))
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
       if frequencies[0]==0:
          frequencies[0]=np.nan
          centroid=np.nan
# fill all frequencies       
       for i in range(len(frequencies)):
          if frequencies[i]==0:
             frequencies[i]=frequencies[0]*(i+1)

# Sum
       allCentroids=np.append(allCentroids,centroid)
       allFrequencies=np.append(allFrequencies,[frequencies],axis=0)
       allMagnitudes=np.append(allMagnitudes,[magnitudes],axis=0)
       allPhases=np.append(allPhases,[phases],axis=0)
       allMfcc=np.append(allMfcc,[mfcc_coeffs],axis=0)
       cnt+=1

   nFrames=cnt

# synthesis
   audioout = []
   for cnt in range(nFrames):
       sfftframe=sinemodelsynth(es.array(allMagnitudes[cnt]),es.array(allFrequencies[cnt]),es.array(allPhases[cnt]))
       ifftframe=ifft(sfftframe)
       out=overlapAdd(ifftframe)
       if cnt >= frameSize / hopSize * 0.2:
           audioout = np.append(audioout,out)

# Write to file
#   audioWriter(audioout.astype(np.float32))

   allMagnitudes[allMagnitudes<-99.9]=np.nan

   f0=allFrequencies[:,0]
# Smooth of basic frequency
   f0_sm=np.mean(f0[~np.isnan(f0)])*np.ones(nFrames)

# Magnitudes to absolute values
   allMagnitudes=10**(allMagnitudes/20.0)

#   print inputFilename,np.mean(f0[~np.isnan(f0)])
# plot basic frequency
   plt.plot(np.arange(nFrames)*hopSize/float(sampleRate),f0,'b')
#   plt.plot(np.arange(nFrames)*hopSize/float(sampleRate),allCentroids,'g')
   plt.plot(np.arange(nFrames)*hopSize/float(sampleRate),f0_sm,'r')
   plt.show()
   f_mean=np.mean(f0[~np.isnan(f0)])

# plot MFCC
   for i in range(1,numberMFCC):
      if i==1:
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

#      plt.plot(np.arange(nFrames)*hopSize/float(sampleRate),allMfcc[:,i],clr)
#   plt.show()
# smooth MFCC

   mf=smooth(allMfcc[:,0],win_len,window='hamming')
   m_mf=np.empty((0,len(mf)))

   mf_mid=[]
   for i in range(numberMFCC):
      mf=smooth(allMfcc[:,i],win_len,window='hamming')
      m_mf=np.append(m_mf,[mf],axis=0)
      mf_mid=np.append(mf_mid,np.mean(m_mf[i]))
   for i in range(1,numberMFCC):
      if i==1:
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

#      plt.plot(np.arange(nFrames)*hopSize/float(sampleRate),allMfcc[:,i],clr)
#      plt.plot(np.arange(nFrames)*hopSize/float(sampleRate),m_mf[i],clr)
#   plt.show()

# spectral analysis of vibrato
   hopSizeVibr=32
   frameSizeVibr=128

   allVibrSp=np.empty((0,frameSizeVibr/2+1))
# frequency resolution for vibrato
   df=float(sampleRate)/hopSize/frameSizeVibr
   cnt=0
# Spectrum
   spectrum_f=Spectrum(size=sampleRate/hopSize)
# SpectralPeaks
   sp_p=SpectralPeaks(sampleRate=sampleRate/hopSize,orderBy='magnitude')
   allFmax=[]
   allAmax=[]
   for frame in FrameGenerator(es.array(f0-f0_sm), frameSize=frameSizeVibr, hopSize=hopSizeVibr, startFromZero=False):
      spect=spectrum_f(window(frame)) 
# find peaks of vibrato's spectrum
      if spect[0]>0:
         sp=sp_p(spect)
         allFmax=np.append(allFmax,sp[0][0]) 
         allAmax=np.append(allAmax,sp[1][0]) 
      else:
         allFmax=np.append(allFmax,np.nan) 
         allAmax=np.append(allAmax,np.nan) 

      cnt+=1
# the peak of vibrato's frequencies
   allFmax=np.mean(allFmax[~np.isnan(allFmax)])
   allAmax=np.mean(allAmax[~np.isnan(allAmax)])
#   print "f_vibr",allFmax
#   print "a_vibr",allAmax
# harmonics
   
#   plt.plot(np.arange(nFrames)*hopSize/float(sampleRate),f0/10-f0_sm/10,'b')
   for i in range(nHarmonics):
      mg=allMagnitudes[:,i]
      if i==0:
        clr='red'
      elif i==1:
        clr='orange'
      elif i==2:
        clr='yellow'
      elif i==3:
        clr='green'
      elif i==4:
        clr='lightblue'
      elif i==5:
        clr='blue'
      elif i==6:
        clr='violet'
      else:
        clr='black'
#      plt.plot(np.arange(nFrames)*hopSize/float(sampleRate),20.0*np.log10(mg),clr)
#   plt.show()
  
# the relation between the frequency and  the amplitude of vibrato
#   plt.scatter(allMagnitudes[:,0],allFrequencies[:,0],color='r')
#   plt.scatter(allMagnitudes[:,1],allFrequencies[:,0],color='orange')
#   plt.scatter(allMagnitudes[:,2],allFrequencies[:,0],color='yellow')
#   plt.scatter(allMagnitudes[:,3],allFrequencies[:,0],color='g')
#   plt.show()

# smooth the magnitudes of the harmonics
   allMagnitudes[np.isnan(allMagnitudes)]=10**(-74/20.0)
   mgn=smooth(allMagnitudes[:,0],win_len,window='hamming')
   m_sm=np.empty((0,len(mgn)))

   for i in range(nHarmonics):
      mgn=smooth(allMagnitudes[:,i],win_len,window='hamming')
      m_sm=np.append(m_sm,[mgn],axis=0)
      if i==0:
        clr='red'
      elif i==1:
        clr='orange'
      elif i==2:
        clr='yellow'
      elif i==3:
        clr='green'
      elif i==4:
        clr='lightblue'
      elif i==5:
        clr='blue'
      elif i==6:
        clr='violet'
      elif i==7:
        clr='cyan'
      else:
        clr='black'
      mg=20.0*np.log10(allMagnitudes[:,i])
      m_sm_n=20.0*(np.log10(m_sm[i]))
#      if True:##i==8:
#        plt.plot(np.arange(nFrames)*hopSize/float(sampleRate),mg,clr)
#        plt.plot(np.arange(nFrames)*hopSize/float(sampleRate),m_sm_n,clr)
#        print i,np.min(mg[~np.isnan(mg)])
#      m_sm_n=m_sm_n[~np.isnan(m_sm_n)]
#      print "harmonic",i+1,m_sm_n
#   plt.show()


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
   for i in range(1,nHarmonics):
       m_sm_n=20.0*(np.log10(m_sm[i]/m_sm[0]))
       mean=np.mean(m_sm_n)
       fp.write("\t"+str(mean))
       m_max=np.max(m_sm[0])
       arg_max=np.argmax(m_sm[0])
       mean=20.0*np.log10(m_sm[i,arg_max]/m_max)
       fp.write("\t"+str(mean))

   fp.write("\n")
   fp.close()
         
nHarmonics=15
numberMFCC=3

outputFile="Adina_RS.dat"
if os.path.exists(outputFile):
    os.remove(outputFile)
   
    
fp=open(outputFile,"w")
fp.write("Filename\tSinger\tPitch_frequency\tMean_frequency")
for i in range(1,nHarmonics):
   fp.write("\tHarm_"+str(i+1)+"\tHarm_p_"+str(i+1))
fp.write("\n")
fp.close()


pt='sounds'
lst=fnmatch.filter(os.listdir(pt), 'RS_Adina_*.wav')


#lst=fnmatch.filter(os.listdir(pt), 'AN_Adina_000_g#-1.wav')
lst.sort()
for f in lst:
   print f
   anal_pitch(pt,f)
