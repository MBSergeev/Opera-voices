import essentia as es
from essentia.standard import *
import numpy as np
import matplotlib.pyplot as plt

def fun():

   sma=SprModelAnal(sampleRate=sampleRate,hopSize=hopSize,fftSize=frameSize, maxnSines=maxnSines,freqDevOffset=freqDevOffset,freqDevSlope=freqDevSlope,magnitudeThreshold=magnitudeThreshold,maxFrequency=maxFrequency,maxPeaks=maxPeaks,minFrequency=minFrequency,orderBy="frequency")

   cnt=0
   audioout1=[]
   audioout2=[]
   audioout3=[]

   for frame in FrameGenerator(audio, frameSize, hopSize, startFromZero=False):
      frequencies,magnitudes,phases,res=sma(frame)
      outframe,sineframe,res=sms(magnitudes,frequencies,phases,res)
      spect=spectrum(window(frame))
      brb=bb(es.array(spect))
      brb=brb/np.sum(brb)/fr_d
      barkbands=10*np.log10(brb)

      if cnt==2000:
          frequencies[frequencies==0]=np.nan
          magnitudes[magnitudes==0]=np.nan
          plt.plot(np.sort(frequencies),magnitudes[np.argsort(frequencies)])
          plt.plot(fr_a,barkbands)
          plt.show()
          return

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

sms=SprModelSynth(sampleRate=sampleRate,hopSize=hopSize,fftSize=frameSize)
window =Windowing(type='hamming')
spectrum=Spectrum(size=frameSize)
bb=BarkBands(sampleRate=sampleRate)

marg=[0,50,100,150,200,300,400,510,630,770,920,1080,1270,1480,1720,2000,2320,2700,3150,3700,4400,5300,6400,7700,9500,12000,15500,20500]#,27000]
fr_a=[]
fr_d=[]
for i in range(len(marg)-1):
   fr_a=np.append(fr_a,(marg[i]+marg[i+1])/2.0)
   fr_d=np.append(fr_d,marg[i+1]-marg[i])


audio=slicer(loader())[0]
freqDevOffset=20
freqDevSlope=0.01
magnitudeThreshold =0
maxFrequency=15000
maxPeaks=100
maxnSines=50
minFrequency=20
fun()



