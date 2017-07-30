import essentia as es
from essentia.standard import *

import numpy as np
import matplotlib.pyplot as plt
import sys

def harm_anal(dir_in_sound,dir_out_sound,inputFilename):

   loader = MonoLoader(filename = dir_in_sound+inputFilename, sampleRate = sampleRate )
   audioWriterOut = MonoWriter(filename=dir_out_sound+inputFilename[0:-5]+"_out.wav")
   audioWriterHarm = MonoWriter(filename=dir_out_sound+inputFilename[0:-5]+"_harm.wav")

   sct=SpectralCentroidTime(sampleRate=sampleRate)
   bb=BarkBands(sampleRate=sampleRate)

   equalLoudness = EqualLoudness()
   predominantMelody=PredominantPitchMelodia(frameSize=frameSize,hopSize=hopSize,sampleRate=sampleRate,\

      filterIterations=3,voicingTolerance=0.2,voiceVibrato=True,minFrequency=80,guessUnvoiced=False)

   hma=HprModelAnal(sampleRate=sampleRate,hopSize=hopSize,fftSize=frameSize,nHarmonics=nHarmonics, maxnSines=maxnSines,freqDevOffset=freqDevOffset,freqDevSlope=freqDevSlope,harmDevSlope=harmDevSlope,magnitudeThreshold=magnitudeThreshold,maxFrequency=maxFrequency,maxPeaks=maxPeaks,minFrequency=minFrequency,stocf=stocf) 

   sms=SprModelSynth(sampleRate=sampleRate,hopSize=hopSize,fftSize=frameSize)
   window =Windowing(type='hamming')
   spectrum=Spectrum(size=frameSize)

   mfcc=MFCC(numberCoefficients=numberMFCC,inputSize=frameSize/2+1)

   k_t=hopSize/float(sampleRate)

   audio=loader()
   eqaudio=equalLoudness(audio)
   pitch,pitchConfidence=predominantMelody(eqaudio)
   cnt=0
   audioout1=[]
   audioout2=[]
   fp=open(outputFilename,"a")

   nFrames=pitch.shape[0]
   for frame in FrameGenerator(audio, frameSize, hopSize, startFromZero=False):
      if cnt%1000==0:
          print cnt,nFrames
      spect=spectrum(window(frame))
      centroid=sct(frame)
      frequencies,magnitudes,phases,res=hma(frame,pitch[cnt])
#      outframe,harmframe,res=sms(magnitudes,frequencies,phases,res)
      for i in range(1,len(frequencies)):
          frequencies[i]=frequencies[0]*(i+1)
      spect1=np.zeros(spect.shape)
      for i in range(len(magnitudes)):
          spect1[int(frequencies[i]*frameSize/sampleRate)]=10**(magnitudes[i]/20)
      barkbands1=bb(es.array(spect1))  
      barkbands=bb(es.array(spect)) 
      mfcc_bands,mfcc_coeffs=mfcc(spect)
      mfcc_bands1,mfcc_coeffs1=mfcc(es.array(spect1))

#      audioout1 = np.append(audioout1,outframe)
#     audioout2 = np.append(audioout2,harmframe)
      fp.write(inputFilename+"\t"+inputFilename[0:2]+"\t"+str(frequencies[0])+"\t"+str(pitch[cnt])+"\t"+str(k_t*cnt)+"\t"+str(centroid))
      for j in range(nHarmonics):
         fp.write("\t"+str(magnitudes[j]))
      for j in range(nHarmonics):
         fp.write("\t"+str(phases[j]))
      for j in range(numberMFCC):
         fp.write("\t"+str(mfcc_coeffs[j]))
      for j in range(numberMFCC):
         fp.write("\t"+str(mfcc_coeffs1[j]))
      for j in range(nBarkband):
         fp.write("\t"+str(barkbands[j]))
      for j in range(nBarkband):
         fp.write("\t"+str(barkbands1[j]))

      fp.write("\n")
      cnt+=1

   fp.close()
#   audioWriterOut(audioout1.astype(np.float32))
#   audioWriterHarm(audioout2.astype(np.float32))

#   nFrames=cnt
#   t_arr=np.arange(nFrames)*hopSize/float(sampleRate)
#   plt.plot(t_arr,allFrequencies[:,0],'b')
#   plt.show()

frameSize = 2*1024
hopSize = 256 
sampleRate = 44100
nHarmonics=15
freqDevOffset=20
freqDevSlope=0.01
harmDevSlope=0.01
magnitudeThreshold =0
maxFrequency=15000
maxPeaks=100
maxnSines=100
minFrequency=20
stocf = 0.2
numberMFCC=16
nBarkband=27


dir_in_sound="../in_sound/"
dir_out_sound="../out_sound/"
dir_data="../data/"

#outputFile="OP_Se_in_ciel"
#outputFile="OP_Crudele"
#outputFile="OP_Bacio"
#outputFile="OP_Ah_non_potrian"
#outputFile="OP_Merce_dilette"
#outputFile="OP_Nachtigall"
#outputFile="OP_Son_Vergin"
#outputFile="OP_O_Rendetemi"
#outputFile="OP_O_legere_hirondelle"
#outputFile="OP_Ouvre_ton_coer"
#outputFile="OP_Villanelle"
outputFile="OP_Spiel_ich"

outputFilename=dir_data+outputFile+".dat"

    
fp=open(outputFilename,"w")

fp.write("Filename\tSinger\tFrequency\tPitch\tTime\tCentroid")
for i in range(nHarmonics):
   fp.write("\tHarm_"+str(i))
for i in range(nHarmonics):
   fp.write("\tPhase_"+str(i))
for i in range(numberMFCC):
   fp.write("\tMFCC_"+str(i))
for i in range(numberMFCC):
   fp.write("\tMFCC1_"+str(i))
for i in range(nBarkband):
   fp.write("\tBarkband_"+str(i))
for i in range(nBarkband):
   fp.write("\tBarkband1_"+str(i))

fp.write("\n")
fp.close()


harm_anal(dir_in_sound,dir_out_sound,outputFile+".flac")
