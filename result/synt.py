import pandas as pd
import essentia as es
from essentia.standard import *
import numpy as np
import matplotlib.pyplot as plt
import sys

hopSize = 256
frameSize = 2*1024
sampleRate = 44100
nHarmonics=15

Filename="OP_bacio.dat"

data=pd.read_table(Filename)

time=data.loc[:,"time"].as_matrix()
fr=data.loc[:,"frequency"].as_matrix()
f0_sm=data.loc[:,"frequency_sm"].as_matrix()

m_lst=[]
for i in range(nHarmonics):   
   m_lst.append("magnitude_sm_"+str(i))
magn_sm=data.loc[:,m_lst].as_matrix()

m_lst=[]
for i in range(nHarmonics):   
   m_lst.append("magnitude_"+str(i))
magn=data.loc[:,m_lst].as_matrix()

fr_arr=[fr[0]]
magn_arr=np.empty((0,nHarmonics))   
magn_arr=np.append(magn_arr,[magn[0]],axis=0)
for i in range(1,time.size):
   for j in range(1,hopSize+1):
      fr_c=fr[i-1]+(fr[i]-fr[i-1])*j/float(hopSize)
      fr_arr=np.append(fr_arr,fr_c)
      mg_c=[]
      for k in range(nHarmonics):
         mg_c=np.append(mg_c,magn[i-1,k]+(magn[i,k]-magn[i-1,k])*j/float(hopSize))
      magn_arr=np.append(magn_arr,[mg_c],axis=0)
   print i,time.size
ph=[0]
for i in range(1,fr_arr.size):
   ph=np.append(ph,(fr_arr[i-1]+fr_arr[i])/2+ph[i-1])
   
audio=(10**(magn_arr[:,0]/20))*np.sin(2*np.pi*ph/float(sampleRate))
for k in range(1,nHarmonics):
   audio+=(10**(magn_arr[:,k]/20))*np.sin(2*np.pi*ph/float(sampleRate)*(k+1))
audioWriter = MonoWriter(filename="ooo.wav")
audioWriter(audio.astype(np.float32))
