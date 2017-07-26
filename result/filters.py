import numpy as np
import essentia as es
from essentia.standard import *
import matplotlib.pyplot as plt

bp=BandPass(bandwidth=50,cutoffFrequency=400)
br=BandReject(bandwidth=50,cutoffFrequency=400)
ap=AllPass(cutoffFrequency=400,order=1,bandwidth=50)
hp=HighPass(cutoffFrequency=400)
lp=LowPass(cutoffFrequency=400)
dc=DCRemoval(cutoffFrequency=400)
mf=MaxFilter(width=100)
audioWriter = MonoWriter(filename="ooo.wav")
sampleRate=44100
dt=0.1
a=1

f0_a=[]
m_a=[]
for f0 in range(200,600):
    audio=a*np.sin(2*np.pi*f0*np.arange(dt*sampleRate)/float(sampleRate))
    audio=dc(es.array(audio))
    m=np.max(audio)
    f0_a=np.append(f0_a,f0)
    m_a=np.append(m_a,m)


plt.plot(f0_a,m_a)
plt.show()    
