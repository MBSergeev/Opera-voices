import numpy as np
import matplotlib.pyplot  as plt
import essentia.standard as ess

M=1024
N=1024
H=512
fs=44100


spectrum=ess.Spectrum(size=N)
window=ess.Windowing(size=M,type="hann")
mfcc=ess.MFCC(numberCoefficients=12,inputSize=N/2+1)
x=ess.MonoLoader(filename="speech-female.wav",sampleRate=fs)()

mfccs=[]

for frame in ess.FrameGenerator(x,frameSize=M, hopSize=H,startFromZero=True):
    mX=spectrum(window(frame))
    mfcc_bands,mfcc_coeffs=mfcc(mX)
    mfccs.append(mfcc_coeffs)
mfccs=np.array(mfccs)
plt.plot(mfccs[:,0])
plt.show()
plt.plot(mfccs[:,1])
plt.show()
