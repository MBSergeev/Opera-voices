import essentia as es
from essentia.standard import *
import numpy as np
import matplotlib.pyplot as plt

def fun():

   hma=HpsModelAnal(sampleRate=sampleRate,hopSize=hopSize,fftSize=frameSize,nHarmonics=nHarmonics, maxnSines=maxnSines,freqDevOffset=freqDevOffset,freqDevSlope=freqDevSlope,harmDevSlope=harmDevSlope,magnitudeThreshold=magnitudeThreshold,maxFrequency=maxFrequency,maxPeaks=maxPeaks,minFrequency=minFrequency,stocf=stocf)

   cnt=0
   audioout1=[]
   audioout2=[]
   audioout3=[]

   for frame in FrameGenerator(audio, frameSize, hopSize, startFromZero=False):
      frequencies,magnitudes,phases,res=hma(frame,pitch[cnt])
      outframe,sineframe,res=sms(magnitudes,frequencies,phases,res)
      if cnt==1000:
          frequencies[frequencies==0]=np.nan
          print frequencies
          magnitudes[magnitudes==-100]=np.nan
          plt.plot(frequencies,magnitudes)
          plt.show()
      audioout1 = np.append(audioout1,sineframe)
      audioout2 = np.append(audioout2,outframe)
      audioout3 = np.append(audioout3,res)
      cnt+=1

   audioWriter1(audioout1.astype(np.float32))
   audioWriter2(audioout2.astype(np.float32))
   audioWriter3(audioout3.astype(np.float32))

   return np.std(audioout3)

frameSize = 2*1024
hopSize = 256 
sampleRate = 44100
nHarmonics=100
maxnSines=100

#inputFilename="OP_bacio1.wav"
inputFilename="OP_Adina.aac"

s_t=117
e_t=205
#s_t=0
#e_t=30
loader = MonoLoader(filename = inputFilename, sampleRate = sampleRate )
slicer=Slicer(startTimes=[s_t],endTimes=[e_t],sampleRate=sampleRate)
audioWriter1 = MonoWriter(filename="ooo1.wav")
audioWriter2 = MonoWriter(filename="ooo2.wav")
audioWriter3 = MonoWriter(filename="ooo3.wav")
equalLoudness = EqualLoudness()

sms=SpsModelSynth(sampleRate=sampleRate,hopSize=hopSize,fftSize=frameSize)
predominantMelody=PredominantPitchMelodia(frameSize=frameSize,hopSize=hopSize,sampleRate=sampleRate, filterIterations=3,voicingTolerance=0.2,voiceVibrato=True,minFrequency=80,guessUnvoiced=False)
audio=slicer(loader())[0]
eqaudio=equalLoudness(audio)
pitch,pitchConfidence=predominantMelody(eqaudio)
freqDevOffset=20
freqDevSlope=0.01
harmDevSlope=0.01
magnitudeThreshold =0
maxFrequency=15000
maxPeaks=100
maxnSines=100
minFrequency=20
nHarmonics=15
stocf = 0.2
print fun()



