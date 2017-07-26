# harmonic model
import essentia as es
from essentia.standard import *

import numpy as np
import matplotlib.pyplot as plt
import sys

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
 

def harm_anal(inputFilename,s_t,e_t):

   outputFilename="sounds1/"+inputFilename[0:-4]+"harm.wav"

   frameSize = 2*1024
   hopSize = 256 
#hopSize = 512 
   sampleRate = 44100
   minF0=200.0
   maxF0=2000.0
   minSineDur = 0.02
   nHarmonics=15
   harmDevSlope=0.01


# functions
   slicer=Slicer(startTimes=[s_t],endTimes=[e_t],sampleRate=sampleRate)
   loader = MonoLoader(filename = inputFilename, sampleRate = sampleRate )
   window =Windowing(type='hamming')
   fft=FFT(size=frameSize)
   spectrum=Spectrum(size=frameSize)
   equalLoudness = EqualLoudness()
   predominantMelody=PredominantPitchMelodia(frameSize=frameSize,hopSize=hopSize,sampleRate=sampleRate)
   pitchDetect = PitchYinFFT(frameSize=frameSize,sampleRate=sampleRate)
   harmonicmodelanal=HarmonicModelAnal(sampleRate=sampleRate,hopSize=hopSize,nHarmonics=nHarmonics,harmDevSlope=harmDevSlope,
       maxFrequency=maxF0,minFrequency=minF0)

   sinemodelsynth = SineModelSynth(sampleRate=sampleRate,fftSize=frameSize,hopSize=hopSize)
   ifft=IFFT(size=frameSize)
   overlapAdd = OverlapAdd(frameSize=frameSize, hopSize=hopSize,  gain=1./frameSize)
   audioWriter = MonoWriter(filename=outputFilename)
   sct=SpectralCentroidTime(sampleRate=sampleRate)
   sc=SpectralComplexity(sampleRate=sampleRate,magnitudeThreshold=0.005)
#vibrato=Vibrato(sampleRate=float(sampleRate)/hopSize,minExtend=1,maxExtend=1000,minFrequency=1,maxFrequency=20)
#read
   audio=slicer(loader())[0]
#audio=loader()
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
   allSpCompl=[]
   cnt=0
# The cycle 
   for frame in FrameGenerator(audio, frameSize, hopSize, startFromZero=False):
# SpectralCentroid
       centroid=sct(frame)
# Spectrum
       spect=spectrum(window(frame))
#SpectralComplexity
       spCompl=sc(spect)
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
#       if i>0:
#            magnitudes[i]=-100.0
             
#       if frequencies[i]>2000:
#           magnitudes[i]=-100.0
       

#    phases[phases!=0]=0

    
# Sum
       allCentroids=np.append(allCentroids,centroid)
       allSpCompl=np.append(allSpCompl,spCompl)
       allFrequencies=np.append(allFrequencies,[frequencies],axis=0)
       allMagnitudes=np.append(allMagnitudes,[magnitudes],axis=0)
       allPhases=np.append(allPhases,[phases],axis=0)
       cnt+=1
#np.set_printoptions(threshold='nan')

   nFrames=cnt

#vibr=vibrato(es.array(allFrequencies[:,0]))
# synthesis
   audioout = []#np.array(0)
   for cnt in range(nFrames):
       sfftframe=sinemodelsynth(es.array(allMagnitudes[cnt]),es.array(allFrequencies[cnt]),es.array(allPhases[cnt]))
       ifftframe=ifft(sfftframe)
       out=overlapAdd(ifftframe)
       if cnt >= frameSize / hopSize * 0.2:
          audioout = np.append(audioout,out)

# Write to file
   audioWriter(audioout.astype(np.float32))
   allMagnitudes[allMagnitudes<-99.9]=np.nan
#allMagnitudes=allMagnitudes+10
# Smooth of basic frequency
   f0_sm=smooth(allFrequencies[:,0],199,window='hamming')

# Magnitudes to absolute values
   allMagnitudes=10**(allMagnitudes/20.0)

   


# plot basic frequency
#   plt.plot(np.arange(nFrames)*hopSize/float(sampleRate),allFrequencies[:,0],'b')
#   plt.plot(np.arange(nFrames)*hopSize/float(sampleRate),f0_sm,'r')
#plt.plot(np.arange(nFrames)*hopSize/float(sampleRate),allCentroids,'g')
#   plt.show()
#  plot the first harmonics
#plt.plot(np.arange(nFrames)*hopSize/float(sampleRate),allMagnitudes[:,0],'b')
#plt.plot(np.arange(nFrames)*hopSize/float(sampleRate),allMagnitudes[:,1],'g')
#plt.plot(np.arange(nFrames)*hopSize/float(sampleRate),allMagnitudes[:,2],'black')
#plt.plot(np.arange(nFrames)*hopSize/float(sampleRate),allMagnitudes[:,3],'r')
#plt.plot(np.arange(nFrames)*hopSize/float(sampleRate),allMagnitudes[:,4],'g')
#plt.plot(np.arange(nFrames)*hopSize/float(sampleRate),allMagnitudes[:,5],'black')
#plt.plot(np.arange(nFrames)*hopSize/float(sampleRate),allMagnitudes[:,6],'r')
#plt.show()

# plot the smooth basic frequency
#plt.plot(np.arange(nFrames)*hopSize/float(sampleRate),allFrequencies[:,0]-f0_sm)
#plt.show()


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
# SpectralCentroid
   sct_v=SpectralCentroidTime(sampleRate=sampleRate/hopSize)
   allCentroidsVibr=[]
   allFmax=[]
   for frame in FrameGenerator(es.array(allFrequencies[:,0]-f0_sm), frameSize=frameSizeVibr, hopSize=hopSizeVibr, startFromZero=False):
      spect=spectrum_f(window(frame)) 
      allVibrSp=np.append(allVibrSp,[spect],axis=0)
      centroid=sct_v(frame)
      allCentroidsVibr=np.append(allCentroidsVibr,centroid)
# find peaks of vibrato's spectrum
      if spect[0]>0:
         sp=sp_p(spect)
         allFmax=np.append(allFmax,sp[0][0]) 
      else:
         allFmax=np.append(allFmax,np.nan) 

      cnt+=1
   
#plt.plot(np.arange(cnt)*hopSizeVibr/float(sampleRate/hopSize),allVibrSp[:,1])
#plt.plot(np.arange(cnt)*hopSizeVibr/float(sampleRate/hopSize),allVibrSp[:,2])
#plt.plot(np.arange(cnt)*hopSizeVibr/float(sampleRate/hopSize),allVibrSp[:,3])
#plt.plot(np.arange(cnt)*hopSizeVibr/float(sampleRate/hopSize),allVibrSp[:,4])
# plot the frequency of vibrato for maximal peak of spectrum
#plt.plot(np.arange(cnt)*hopSizeVibr/float(sampleRate/hopSize),allFmax)
#plt.plot(np.arange(cnt)*hopSizeVibr/float(sampleRate/hopSize),allCentroidsVibr)
#plt.show()
# smooth the magnitudes of the harmonics
   mgn=smooth(allMagnitudes[:,0],99,window='hamming')
   m_sm=np.empty((0,len(mgn)))

   for i in range(15):
      mgn=smooth(allMagnitudes[:,i],99,window='hamming')
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
      else:
        clr='black'
#      plt.plot(np.arange(nFrames)*hopSize/float(sampleRate),20.0*np.log10(m_sm[i]),clr)
#   plt.plot(np.arange(nFrames)*hopSize/float(sampleRate),f0_sm/10,'r')

#   plt.show()
# The corellation of the main frequency and the first harmonic
#f0=[]
#m0=[]
#for d in  zip(allFrequencies[:,0]-f0_sm,allMagnitudes[:,0]-m0_sm):
#   if not np.isnan(d[0]) and not np.isnan(d[1]):
#     f0=np.append(f0,d[0])
#     m0=np.append(m0,d[1])

#plt.scatter(f0,m0)
#plt.show()

# ratio of smoothed magnitudes
   """
m_sm=m_sm/m_sm[0]

for i in range(15):
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

   plt.plot(f0_sm,20.0*np.log10(m_sm[i]),clr)
plt.show()

   """

# Find long pitches

   isPitch=False
   delta_f=2**(1/12.0)
   delta_t=0.4
   acc_pitch=[]
   for i in range(len(f0_sm)):
      f=f0_sm[i]
#      print i,f,get_score(f)
      if not isPitch:
         if not np.isnan(f):
           f_start=f
           pitch=get_score(f_start)
           i_start=i
           isPitch=True
      else:
         if np.isnan(f):
           f_end=f0_sm[i-1]
           i_end=i-1
           acc_pitch.append([[i_start,f_start,pitch],[i_end,f_end,pitch]])
           isPitch=False
         else:
           if get_score(f)!=pitch:
                f_end=f0_sm[i-1]
                i_end=i-1
                acc_pitch.append([[i_start,f_start,pitch],[i_end,f_end,pitch]])
                f_start=f0_sm[i]
                pitch=get_score(f_start)
                i_start=i
   if isPitch:
      f_end=f0_sm[i-1]
      i_end=i-1
      acc_pitch.append([[i_start,f_start,pitch],[i_end,f_end,pitch]])

   t0_l=[]
   t1_l=[]
   acc_pitch_s=[]
   for p in acc_pitch:
      t0=p[0][0]/float(sampleRate)*hopSize
      t1=p[1][0]/float(sampleRate)*hopSize
      dt=t1-t0
#      print p,dt
      if dt>delta_t:
#          print p,dt
          t0_l.append(t0+s_t)
          t1_l.append(t1+s_t)
          acc_pitch_s.append(p)
         
   slicer=Slicer(startTimes=t0_l,endTimes=t1_l,sampleRate=sampleRate)
   audio=slicer(loader())
   cnt=0
   for p in acc_pitch_s:
#      print cnt,acc_pitch_s[cnt],t0_l[cnt],t1_l[cnt]
      audioWriter = MonoWriter(filename="sounds1/"+inputFilename[0:-4]+"_"+"%03d" % (cnt,)+"_"+p[0][2][0]+".wav")
      audio_c=audio[cnt]
      audioWriter(audio_c)
      cnt+=1


harm_anal('AN_Adina.wav',97,192)
#harm_anal('OP_Adina.aac',106,205)
#harm_anal('VN_Adina.aac',105,193)
#harm_anal('LP_Adina.wav',0,82)
#harm_anal('MD_Adina.wav',0,93)
#harm_anal('JS_Adina.wav',0,80)
#harm_anal('AG_Adina.wav',0,84)
#harm_anal('AK_Adina.wav',0,82)
#harm_anal('LC_Adina.wav',0,71)
#harm_anal('AF_Adina.wav',0,81)
#harm_anal('IC_Adina.wav',0,85)
#harm_anal('KB_Adina.wav',0,86)
#harm_anal('ES_Adina.wav',0,76)
#harm_anal('MC_Adina.wav',0,65)
#harm_anal('RS_Adina.wav',0,89)
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
# MC - Maria Callas
# RS - Renata Scotto
