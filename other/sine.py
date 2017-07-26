# sin-model
import essentia
import essentia.standard
from essentia.standard import *

import numpy as np

inputFilename = 'singing-female.wav'
outputFilename='singing-female2.wav'
frameSize = 2048
hopSize = 512 
sampleRate = 44100
maxnSines=100
magnitudeThreshold=-74
minSineDur=0.02
freqDevOffset = 10
freqDevSlope=0.001

loader = MonoLoader(filename = inputFilename, sampleRate = sampleRate )
audio=loader()

fcut = FrameCutter(frameSize = frameSize, hopSize = hopSize, startFromZero=False)
#w = Windowing(type = "blackmanharris92")
w = Windowing(type = "hamming")
fft = FFT(size = frameSize)
ifft = IFFT(size = frameSize);
overl = OverlapAdd (frameSize = frameSize, hopSize = hopSize, gain = 1./frameSize);
awrite = MonoWriter (filename = outputFilename, sampleRate = sampleRate);
smsyn = SineModelSynth(sampleRate = sampleRate, fftSize = frameSize, hopSize = hopSize)
smanal = SineModelAnal(sampleRate = sampleRate, maxnSines = maxnSines, magnitudeThreshold = magnitudeThreshold, 
freqDevOffset = freqDevOffset, freqDevSlope = freqDevSlope)

audioout = np.array(0)

for frame in FrameGenerator(audio, frameSize = frameSize, hopSize = hopSize):
    infft = fft(w(frame))
    magnitudes,frequencies,phases=smanal(infft)
    outfft=smsyn(frequencies,magnitudes,phases)
    
    out=overl(ifft(outfft))
    audioout = np.append(audioout, out)


awrite(audioout.astype(np.float32))

