#stft analyse-synthese
import essentia
import essentia.standard
from essentia.standard import *

import numpy as np

inputFilename = 'singing-female.wav'
outputFilename='singing-female1.wav'
frameSize = 2048
hopSize = 128 
sampleRate = 44100.0
loader = MonoLoader(filename = inputFilename, sampleRate = sampleRate )
audio=loader()

fcut = FrameCutter(frameSize = frameSize, hopSize = hopSize, startFromZero=False)
w = Windowing(type = "hann")
fft = FFT(size = frameSize)
ifft = IFFT(size = frameSize);
overl = OverlapAdd (frameSize = frameSize, hopSize = hopSize, gain = 1./frameSize);
awrite = MonoWriter (filename = outputFilename, sampleRate = 44100);

audioout = np.array(0)

for frame in FrameGenerator(audio, frameSize = frameSize, hopSize = hopSize):
    infft = fft(w(frame))
    outfft=infft
    out=overl(ifft(outfft))
    audioout = np.append(audioout, out)


awrite(audioout.astype(np.float32))

