import numpy as np
import matplotlib.pyplot  as plt
from essentia.standard import *

frameSize=1024
hopSize=256
sampleRate=44100

inputFilename='OP_bacio.wav'

spectrum=Spectrum(size=frameSize)
window=Windowing(size=frameSize,type="hann")
mfcc=MFCC(numberCoefficients=12,inputSize=frameSize/2+1)
equalLoudness = EqualLoudness()
predominantMelody=PredominantPitchMelodia(frameSize=frameSize,hopSize=hopSize,sampleRate=sampleRate)
slicer=Slicer(startTimes=[0],endTimes=[22],sampleRate=sampleRate)
loader = MonoLoader(filename = inputFilename, sampleRate = sampleRate )


audio=slicer(loader())[0]

eqaudio=equalLoudness(audio)


pitch,pitchConfidence=predominantMelody(eqaudio)



mfccs=[]
nFrames=0
for frame in FrameGenerator(audio,frameSize=frameSize, hopSize=hopSize,startFromZero=False):
    mX=spectrum(window(frame))
    mfcc_bands,mfcc_coeffs=mfcc(mX)
    mfccs.append(mfcc_coeffs)
    nFrames+=1
mfccs=np.array(mfccs)

i=0
for p in pitch:
  if p==0:
     pitch[i]=None
     mfccs[i,1]=None
  i+=1

plt.plot(np.arange(nFrames)*hopSize/float(sampleRate),mfccs[:,1],'b')
plt.plot(np.arange(nFrames)*hopSize/float(sampleRate),mfccs[:,2],'g')
plt.plot(np.arange(nFrames)*hopSize/float(sampleRate),mfccs[:,3],'r')
plt.plot(np.arange(nFrames)*hopSize/float(sampleRate),mfccs[:,4],'black')
plt.show()
print len(np.arange(nFrames)*hopSize/float(sampleRate)),len(pitch)
plt.plot(np.arange(nFrames)*hopSize/float(sampleRate),pitch)
plt.show()

pitch_n=[]
mfccs1=[]
mfccs2=[]
mfccs3=[]
mfccs4=[]
mfccs5=[]
for p in enumerate(pitch):
    if not np.isnan(p[1]) and not np.isnan(mfccs[p[0],1]) and not np.isnan(mfccs[p[0],2]) :
       pitch_n.append(p[1])
       mfccs1.append(mfccs[p[0],1])
       mfccs2.append(mfccs[p[0],2])
       mfccs3.append(mfccs[p[0],3])
       mfccs4.append(mfccs[p[0],4])
       mfccs5.append(mfccs[p[0],5])
#for n in  zip(pitch_n,mfccs1):
#   print n
plt.plot(pitch_n,mfccs1,'b')
plt.plot(pitch_n,mfccs2,'g')
#plt.plot(pitch_n,mfccs4,'r')
plt.plot(pitch_n,mfccs5,'black')
plt.show()
