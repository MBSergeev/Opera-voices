# sine_substraction-model
import essentia
import essentia.standard
from essentia.standard import *

import numpy as np

inputFilename = 'singing-female.wav'
outputFilename='singing-female3.wav'
frameSize = 2048
hopSize = 128 
sampleRate = 44100
maxnSines=100
magnitudeThreshold=-74
minSineDur=0.02
freqDevOffset = 10
freqDevSlope=0.001

loader = MonoLoader(filename = inputFilename, sampleRate = sampleRate )
audio=loader()

fcut = FrameCutter(frameSize = frameSize, hopSize = hopSize, startFromZero=False)
w = Windowing(type = "blackmanharris92")
fft = FFT(size = frameSize)
awrite = MonoWriter (filename = outputFilename, sampleRate = sampleRate);
smanal = SineModelAnal(sampleRate = sampleRate, maxnSines = maxnSines, magnitudeThreshold = magnitudeThreshold, 
freqDevOffset = freqDevOffset, freqDevSlope = freqDevSlope)
subtrFFTSize = min(frameSize/4, 4*hopSize)
smsub = SineSubtraction(sampleRate = sampleRate, fftSize = subtrFFTSize, hopSize = hopSize)

pool=essentia.Pool()
for frame in FrameGenerator(audio, frameSize = frameSize, hopSize = hopSize):
    infft = fft(w(frame))
    magnitudes,frequencies,phases=smanal(infft)
    out=smsub(frame,frequencies,magnitudes,phases)
    pool.add('frames',out)

audioout = pool['frames'].flatten()

awrite(audioout.astype(np.float32))


