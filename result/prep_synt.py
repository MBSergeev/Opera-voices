import essentia as es
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
    return min_f


def smooth(x,window_len=11,window='hamming'):
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


def prep_data(inputFilename,s_t,e_t):
   loader = MonoLoader(filename = inputFilename, sampleRate = sampleRate )
   slicer=Slicer(startTimes=[s_t],endTimes=[e_t],sampleRate=sampleRate)
   audioWriterOut = MonoWriter(filename=inputFilename[0:-4]+"_out.wav")
   audioWriterHarm = MonoWriter(filename=inputFilename[0:-4]+"_harm.wav")
   audioWriterRes = MonoWriter(filename=inputFilename[0:-4]+"_res.wav")


   hma=HprModelAnal(sampleRate=sampleRate,hopSize=hopSize,fftSize=frameSize,nHarmonics=nHarmonics, maxnSines=maxnSines,freqDevOffset=freqDevOffset,freqDevSlope=freqDevSlope,harmDevSlope=harmDevSlope,magnitudeThreshold=magnitudeThreshold,maxFrequency=maxFrequency,maxPeaks=maxPeaks,minFrequency=minFrequency,stocf=stocf)

   sms=SprModelSynth(sampleRate=sampleRate,hopSize=hopSize,fftSize=frameSize)
   equalLoudness = EqualLoudness()
   predominantMelody=PredominantPitchMelodia(frameSize=frameSize,hopSize=hopSize,sampleRate=sampleRate, filterIterations=3,voicingTolerance=0.2,voiceVibrato=True,minFrequency=80,guessUnvoiced=False)

   audio=slicer(loader())[0]
   eqaudio=equalLoudness(audio)
   pitch,pitchConfidence=predominantMelody(eqaudio)

   cnt=0
   audioout1=[]
   audioout2=[]
   audioout3=[]
   k_t=hopSize/float(sampleRate)
   allFrequencies=np.empty((0,nHarmonics))
   allMagnitudes=np.empty((0,nHarmonics))
   allPhases=np.empty((0,nHarmonics))
   allRes=np.empty((0,hopSize))
   
   for frame in FrameGenerator(audio, frameSize, hopSize, startFromZero=False):
#      if cnt%1000==0:
#          print cnt
      frequencies,magnitudes,phases,res=hma(frame,pitch[cnt])
      if frequencies[0]==0:
        continue
      allFrequencies=np.append(allFrequencies,[frequencies],axis=0)
      allMagnitudes=np.append(allMagnitudes,[magnitudes],axis=0)
      allPhases=np.append(allPhases,[phases],axis=0)
      allRes=np.append(allRes,[res],axis=0)
      cnt+=1

   nFrames=cnt
   k_t=hopSize/float(sampleRate)
   f0_sm=smooth(allFrequencies[:,0],69)
   """
   plt.plot(np.arange(nFrames)*k_t,allFrequencies[:,0])
   plt.plot(np.arange(nFrames)*k_t,f0_sm)
   for i in range(-12,16):
      fr_t=2**(i/12.0)*440
      plt.plot(np.arange(nFrames)*k_t,fr_t*np.ones(nFrames),'black')
      plt.text(0,fr_t,get_score(fr_t)[0])
   plt.show()

#   plt.plot(np.arange(nFrames)*k_t,allPhases[:,1])
#   plt.show()
   autocorr=np.correlate(allPhases[:,1],allPhases[:,1],mode='full')
   autocorr=autocorr[autocorr.size/2:]
   autocorr=autocorr/autocorr[0]
#   plt.plot(autocorr)
#   plt.show()
    """
   

      
   allMagnitudes[allMagnitudes==-100]=-74
   allMagnSm=np.empty((0,nFrames))
   for i in range(nHarmonics):
      magn_sm=smooth(allMagnitudes[:,i],69)
      allMagnSm=np.append(allMagnSm,[magn_sm],axis=0)
    
   allMagnSm=allMagnSm.transpose() 
   """
   for k in range(0,9):
      plt.plot(np.arange(nFrames)*k_t,allMagnitudes[:,k])
      plt.plot(np.arange(nFrames)*k_t,allMagnSm[:,k])
   plt.show()
   """
   fp=open(inputFilename[0:-4]+".dat","w")
   fp.write("time\tfrequency\tfrequency_sm")
   for j in range(nHarmonics):
      fp.write("\tmagnitude_"+str(j))
   for j in range(nHarmonics):
      fp.write("\tmagnitude_sm_"+str(j))
   fp.write("\n")
   for i in range(nFrames):
      magnitudes=es.array(allMagnitudes[i])
      frequencies=es.array(allFrequencies[i])
      phases=es.array(allPhases[i])
      res=es.array(allRes[i])

      magnitudes=es.array(allMagnSm[i])
#      frequencies[:]=0 
#      frequencies[0]=f0_sm[i]
#      for j in range(1,nHarmonics):
#         frequencies[j]=frequencies[0]*(j+1)
#      phases[:]=np.random.random()*2*np.pi-np.pi
     
      outframe,harmframe,res=sms(magnitudes,frequencies,phases,res)
      audioout1 = np.append(audioout1,outframe)
      audioout2 = np.append(audioout2,harmframe)
      audioout3 = np.append(audioout3,res)

      fp.write(str(i*k_t)+"\t"+str(allFrequencies[i,0])+"\t"+str(f0_sm[i]))
      for j in range(0,nHarmonics):
        fp.write("\t"+str(allMagnitudes[i,j]))
      for j in range(0,nHarmonics):
        fp.write("\t"+str(allMagnSm[i,j]))
      fp.write("\n")
      
   fp.close()

   audioWriterOut(audioout1.astype(np.float32))
   audioWriterHarm(audioout2.astype(np.float32))
   audioWriterRes(audioout3.astype(np.float32))


frameSize = 2*1024
hopSize = 256
sampleRate = 44100
nHarmonics=15
freqDevOffset=20
freqDevSlope=0.01
harmDevSlope=0.01
magnitudeThreshold =0
maxFrequency=15000
maxPeaks=100
maxnSines=100
minFrequency=20
stocf = 0.2

prep_data("OP_bacio.wav",2.1,2.7)
#prep_data("OP_bacio.wav",1.5,8)

