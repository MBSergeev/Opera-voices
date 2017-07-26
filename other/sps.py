# sps-model
import essentia
import essentia.standard
import matplotlib.pyplot as plt
from essentia.standard import *

import numpy as np

inputFilename = 'singing-female.wav'
outputFilename='singing-female5.wav'
#inputFilename = 'flamenco.wav'
#outputFilename='flamenco5.wav'
frameSize = 2048
hopSize = 128 
sampleRate = 44100
maxnSines=5
magnitudeThreshold=-74
minSineDur=0.02
freqDevOffset = 10
freqDevSlope=0.001
stocf=0.2

loader = MonoLoader(filename = inputFilename, sampleRate = sampleRate )
audio=loader()

#fcut = FrameGenerator(frameSize = frameSize, hopSize = hopSize, startFromZero=False)
#w = Windowing(type = "blackmanharris92")
awrite = MonoWriter (filename = outputFilename, sampleRate = sampleRate);
smanal = SpsModelAnal(sampleRate = sampleRate, maxnSines = maxnSines, stocf=stocf, freqDevOffset = freqDevOffset, freqDevSlope = freqDevSlope)
#synFFTSize = min(frameSize/4, 4*hopSize)

smsyn = SpsModelSynth(sampleRate = sampleRate, fftSize = frameSize, hopSize = hopSize, stocf=stocf)

pool=essentia.Pool()
i=0
for frame in FrameGenerator(audio, frameSize = frameSize, hopSize = hopSize, startFromZero=False):
    magnitudes,frequencies,phases,res=smanal(frame)
    rrr=smanal(frame)
    if i==200:
       print rrr
    out,sineframe,resframe=smsyn(frequencies,magnitudes,phases,res)
    pool.add('frames',out)
    pool.add('sine',sineframe)
    pool.add('res',res)
    i+=1

plt.plot(pool['frames'].flatten(),'b')
plt.plot(pool['sine'].flatten(),'g')
#plt.plot(pool['res'].flatten(),'r')
plt.show()
audioout = pool['frames'].flatten()

awrite(audioout.astype(np.float32))


