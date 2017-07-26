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

   frameSize = 2*1024
   hopSize = 256 
#hopSize = 512 
   sampleRate = 44100
   minF0=200.0
   maxF0=1000.0
   minSineDur = 0.02
   nHarmonics=15
   harmDevSlope=0.01
   numberMFCC=12
   window_len=69


   slicer=Slicer(startTimes=[s_t],endTimes=[e_t],sampleRate=sampleRate)
   loader = MonoLoader(filename = inputFilename, sampleRate = sampleRate )
   window =Windowing(type='hamming')
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
   outputFilename=inputFilename[0:-4]+"_out.wav"
   audioWriter = MonoWriter(filename=outputFilename)

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
   cnt=0
   for frame in FrameGenerator(audio, frameSize, hopSize, startFromZero=False):
      spect=spectrum(window(frame))
      infft = fft(window(frame))
      frequencies,magnitudes,phases=harmonicmodelanal(infft,pitch[cnt])
      if frequencies[0]==0:
           frequencies.fill(np.nan)
      
      allFrequencies=np.append(allFrequencies,[frequencies],axis=0)
      allMagnitudes=np.append(allMagnitudes,[magnitudes],axis=0)
      allPhases=np.append(allPhases,[phases],axis=0)
      cnt+=1


   f0_sm=smooth(allFrequencies[:,0],window_len,window='hamming')

   nFrames=cnt

# synthesis
   audioout = []
   print nFrames
   for cnt in range(nFrames):
       if cnt%1000==0:
          print cnt
       sfftframe=sinemodelsynth(es.array(allMagnitudes[cnt]),es.array(allFrequencies[cnt]),es.array(allPhases[cnt]))
       ifftframe=ifft(sfftframe)
       out=overlapAdd(ifftframe)
       if cnt >= frameSize / hopSize * 0.1:
          audioout = np.append(audioout,out)

# Write to file
   audioWriter(audioout.astype(np.float32))

   isPitch=False
   delta_f=2**(1/12.0)
   delta_t=0.4
   acc_pitch=[]
   for i in range(len(f0_sm)):
      f=f0_sm[i]
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
      if dt>delta_t:
#          print p,dt
          t0_l.append(t0+s_t)
          t1_l.append(t1+s_t)
          acc_pitch_s.append([p,t0,t1])

   t_arr=np.arange(nFrames)*hopSize/float(sampleRate)
   plt.plot(t_arr,allFrequencies[:,0],'b')
   plt.plot(t_arr,f0_sm,'r')
#   plt.plot(t_arr,pitch,'r')
#   for i in range(-12,16):
   for i in range(-18,5):
      fr_t=2**(i/12.0)*440
      plt.plot(t_arr,fr_t*np.ones(len(t_arr)),'black')
      plt.text(0,fr_t,get_score(fr_t)[0])
   for p in acc_pitch_s:
      plt.plot([p[1],p[1]],[0,1200],'g')
      plt.plot([p[2],p[2]],[0,1200],'r')
      plt.text(p[1],10,p[0][0][2][0])
   plt.show()
   slicer=Slicer(startTimes=t0_l,endTimes=t1_l,sampleRate=sampleRate)

   audio=slicer(loader())
   cnt=0
   for p in acc_pitch_s:
      audioWriter = MonoWriter(filename="sounds/"+inputFilename[0:-4]+"_"+"%03d" % (cnt,)+"_"+p[0][0][2][0]+".wav")
      audio_c=audio[cnt]
      audioWriter(audio_c)
      cnt+=1


harm_anal('zykina.m4a',0,134)
