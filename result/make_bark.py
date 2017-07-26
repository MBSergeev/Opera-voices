import essentia as es
from essentia.standard import *

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

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

   print inputFilename

   frameSize = 2*1024
   hopSize = 256 
#hopSize = 512 
   sampleRate = 44100
   minF0=200.0
   maxF0=2000.0
   minSineDur = 0.02
   harmDevSlope=0.01
   numberMFCC=12
   window_len=199


   slicer=Slicer(startTimes=[s_t],endTimes=[e_t],sampleRate=sampleRate)
   loader = MonoLoader(filename = inputFilename, sampleRate = sampleRate )
   window =Windowing(type='hamming')
   sct=SpectralCentroidTime(sampleRate=sampleRate)
   fft=FFT(size=frameSize)
   spectrum=Spectrum(size=frameSize)
   equalLoudness = EqualLoudness()
   predominantMelody=PredominantPitchMelodia(frameSize=frameSize,hopSize=hopSize,sampleRate=sampleRate,\
      filterIterations=3,voicingTolerance=0.2,voiceVibrato=True,minFrequency=80,guessUnvoiced=False)
   harmonicmodelanal=HarmonicModelAnal(sampleRate=sampleRate,hopSize=hopSize,nHarmonics=nHarmonics,harmDevSlope=harmDevSlope,
       maxFrequency=maxF0,minFrequency=minF0)

   sinemodelsynth = SineModelSynth(sampleRate=sampleRate,fftSize=frameSize,hopSize=hopSize)
   ifft=IFFT(size=frameSize)
   overlapAdd = OverlapAdd(frameSize=frameSize, hopSize=hopSize,  gain=1./frameSize)
   bb=BarkBands(sampleRate=sampleRate)
   audioWriter = MonoWriter(filename="out2.wav"
)

   audio=slicer(loader())[0]
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
   allBarkbands=np.empty((0,nBarkband))
   allBarkbands1=np.empty((0,nBarkband))
   cnt=0
   for frame in FrameGenerator(audio, frameSize, hopSize, startFromZero=False):
      centroid=sct(frame)
      spect=spectrum(window(frame))
      infft = fft(window(frame))
      frequencies,magnitudes,phases=harmonicmodelanal(infft,pitch[cnt])
      if frequencies[0]==0:
           frequencies.fill(np.nan)
           magnitudes.fill(np.nan)
           phases.fill(np.nan)
           centroid=np.nan
# fill all frequencies       
      else:
           for i in range(1,len(frequencies)):
              frequencies[i]=frequencies[0]*(i+1)
           magnitudes[magnitudes<-99.9]=np.nan
      spect1=np.zeros(spect.shape)
      for i in range(len(magnitudes)):
         if not np.isnan(frequencies[i]) and not np.isnan(magnitudes[i]):
#             print cnt,i,pitch[cnt],frequencies[i],frequencies[i]*frameSize/sampleRate,magnitudes[i]
            spect1[int(frequencies[i]*frameSize/sampleRate)]=10**(magnitudes[i]/20)
      barkbands1=bb(es.array(spect1))  
      barkbands=bb(es.array(spect))  
      magnitudes[np.isnan(magnitudes)] = -74
      allCentroids=np.append(allCentroids,centroid)
      allFrequencies=np.append(allFrequencies,[frequencies],axis=0)
      allMagnitudes=np.append(allMagnitudes,[magnitudes],axis=0)
      allPhases=np.append(allPhases,[phases],axis=0)
      allBarkbands=np.append(allBarkbands,[barkbands[0:nBarkband]],axis=0)
      allBarkbands1=np.append(allBarkbands1,[barkbands1[0:nBarkband]],axis=0)
      cnt+=1

   f0_sm=smooth(allFrequencies[:,0],99,window='hamming')

   nFrames=cnt


   t_arr=np.arange(nFrames)*hopSize/float(sampleRate)
   plt.plot(t_arr,allFrequencies[:,0],'b')
   plt.plot(t_arr,f0_sm,'r')
   for i in range(-12,16):
      fr_t=2**(i/12.0)*440
      plt.plot(t_arr,fr_t*np.ones(len(t_arr)),'black')
      plt.text(0,fr_t,get_score(fr_t)[0])

#   plt.show()

#   for i in range(nHarmonics):
   plt.plot(t_arr,f0_sm/10,"black")
   mg0=smooth(allMagnitudes[:,0],window_len,window='hamming')
   mg0[np.isnan(f0_sm)]=np.nan
   plt.plot(t_arr,mg0,"black")
   allMgn=np.empty((0,nFrames))
   allMgn=np.append(allMgn,[mg0],axis=0)
   for i in range(1,nHarmonics):
      mg=allMagnitudes[:,i]-allMagnitudes[:,0]
      mgn=smooth(allMagnitudes[:,i]-allMagnitudes[:,0],window_len,window='hamming')
      mgn[np.isnan(f0_sm)]=np.nan
      allMgn=np.append(allMgn,[mgn],axis=0)
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
      plt.plot(t_arr,mgn,color=clr)
#   plt.show()
   """
# synthesis
   audioout = []
   print nFrames
   for cnt in range(nFrames):
       if cnt%1000==0:
          print cnt
       if np.isnan(f0_sm[cnt]):
#          continue    
          allMagnitudes[cnt]=np.nan
          allFrequencies[cnt]=np.nan
          allPhases[cnt]=np.nan
       sfftframe=sinemodelsynth(es.array(allMagnitudes[cnt]),es.array(allFrequencies[cnt]),es.array(
allPhases[cnt]))
       ifftframe=ifft(sfftframe)
       out=overlapAdd(ifftframe)
       if cnt >= frameSize / hopSize * 0.1:
          audioout = np.append(audioout,out)

# Write to file
   audioWriter(audioout.astype(np.float32))

   """

   fp=open(outputFilename,"a")
   for i in range(nFrames):
         fp.write(inputFilename+"\t"+inputFilename[0:2]+"\t"+str(f0_sm[i])+"\t"+str(allFrequencies[i,0])+"\t"+str(pitch[i])+"\t"+str(t_arr[i]))
         for j in range(nHarmonics):
            fp.write("\t"+str(allMgn[j][i]))
         for j in range(nBarkband):
            fp.write("\t"+str(allBarkbands[i][j]))
         for j in range(nBarkband):
            fp.write("\t"+str(allBarkbands1[i][j]))
         fp.write("\n")
   fp.close()
       
nHarmonics=15
nBarkband=27


outputFilename="Adina_bark.dat"
if os.path.exists(outputFilename):
    os.remove(outputFilename)
    
fp=open(outputFilename,"w")

fp.write("Filename\tSinger\tMean_frequency\tFrequency\tPitch\tTime\tHarm_0")
for i in range(1,nHarmonics):
   fp.write("\tHarm_"+str(i))
for i in range(nBarkband):
   fp.write("\tBarkBand_"+str(i))
for i in range(nBarkband):
   fp.write("\tBarkBand1_"+str(i))
fp.write("\n")
fp.close()



harm_anal('OP_Adina.aac',117,205)
harm_anal('AN_Adina.wav',97,192)
harm_anal('VN_Adina.aac',105,193)
harm_anal('LP_Adina.wav',0,82)
harm_anal('MD_Adina.wav',0,93)
harm_anal('JS_Adina.wav',0,80)
harm_anal('AG_Adina.wav',0,82)
harm_anal('AK_Adina.wav',0,82)
harm_anal('LC_Adina.wav',0,71)
harm_anal('AF_Adina.wav',0,81)
harm_anal('IC_Adina.wav',0,85)
harm_anal('KB_Adina.wav',0,86)
harm_anal('ES_Adina.wav',0,76)
harm_anal('RS_Adina.wav',0,89)

