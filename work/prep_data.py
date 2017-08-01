import essentia as es
from essentia.standard import *

import numpy as np
import matplotlib.pyplot as plt
import sys

from my_cons import my_cons

def harm_anal(dir_in_sound,dir_out_sound,inputFilename,outputFilename):

   loader = MonoLoader(filename = my_c.dir_in_sound+inputFilename, sampleRate = my_c.sampleRate )
   audioWriterOut = MonoWriter(filename=dir_out_sound+inputFilename[0:-5]+"_out.wav")
   audioWriterHarm = MonoWriter(filename=dir_out_sound+inputFilename[0:-5]+"_harm.wav")

   sct=SpectralCentroidTime(sampleRate=my_c.sampleRate)
   bb=BarkBands(sampleRate=my_c.sampleRate)

   equalLoudness = EqualLoudness()
   predominantMelody=PredominantPitchMelodia(frameSize=my_c.frameSize,hopSize=my_c.hopSize,sampleRate=my_c.sampleRate,\
      filterIterations=3,voicingTolerance=0.2,voiceVibrato=True,minFrequency=80,guessUnvoiced=False)

   hma=HprModelAnal(sampleRate=my_c.sampleRate,hopSize=my_c.hopSize,fftSize=my_c.frameSize,nHarmonics=my_c.nHarmonics, \
      maxnSines=my_c.maxnSines,freqDevOffset=my_c.freqDevOffset,freqDevSlope=my_c.freqDevSlope,harmDevSlope=my_c.harmDevSlope,\
      magnitudeThreshold=my_c.magnitudeThreshold,maxFrequency=my_c.maxFrequency,maxPeaks=my_c.maxPeaks,\
      minFrequency=my_c.minFrequency,stocf=my_c.stocf) 

   sms=SprModelSynth(sampleRate=my_c.sampleRate,hopSize=my_c.hopSize,fftSize=my_c.frameSize)
   window =Windowing(type='hamming')
   spectrum=Spectrum(size=my_c.frameSize)

   mfcc=MFCC(numberCoefficients=my_c.numberMFCC,inputSize=my_c.frameSize/2+1)

   k_t=my_c.hopSize/float(my_c.sampleRate)

   audio=loader()
   eqaudio=equalLoudness(audio)
   pitch,pitchConfidence=predominantMelody(eqaudio)
   cnt=0
   audioout1=[]
   audioout2=[]
   fp=open(outputFilename,"a")

   nFrames=pitch.shape[0]
   for frame in FrameGenerator(audio, my_c.frameSize, my_c.hopSize, startFromZero=False):
      if cnt%10000==0:
          print cnt,nFrames
      spect=spectrum(window(frame))
      centroid=sct(frame)
      frequencies,magnitudes,phases,res=hma(frame,pitch[cnt])
#      outframe,harmframe,res=sms(magnitudes,frequencies,phases,res)
      for i in range(1,len(frequencies)):
          frequencies[i]=frequencies[0]*(i+1)
      spect1=np.zeros(spect.shape)
      for i in range(len(magnitudes)):
          spect1[int(frequencies[i]*my_c.frameSize/my_c.sampleRate)]=10**(magnitudes[i]/20)
      barkbands1=bb(es.array(spect1))  
      barkbands=bb(es.array(spect)) 
      mfcc_bands,mfcc_coeffs=mfcc(spect)
      mfcc_bands1,mfcc_coeffs1=mfcc(es.array(spect1))

#      audioout1 = np.append(audioout1,outframe)
#     audioout2 = np.append(audioout2,harmframe)
      fp.write(inputFilename+"\t"+inputFilename[0:2]+"\t"+str(frequencies[0])+"\t"+str(pitch[cnt])+"\t"+str(k_t*cnt)+"\t"+str(centroid))
      for j in range(my_c.nHarmonics):
         fp.write("\t"+str(magnitudes[j]))
      for j in range(my_c.nHarmonics):
         fp.write("\t"+str(phases[j]))
      for j in range(my_c.numberMFCC):
         fp.write("\t"+str(mfcc_coeffs[j]))
      for j in range(my_c.numberMFCC):
         fp.write("\t"+str(mfcc_coeffs1[j]))
      for j in range(my_c.nBarkband):
         fp.write("\t"+str(barkbands[j]))
      for j in range(my_c.nBarkband):
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

def write_head(outputFilename):

   fp=open(outputFilename,"w")

   fp.write("Filename\tSinger\tFrequency\tPitch\tTime\tCentroid")
   for i in range(my_c.nHarmonics):
      fp.write("\tHarm_"+str(i))
   for i in range(my_c.nHarmonics):
      fp.write("\tPhase_"+str(i))
   for i in range(my_c.numberMFCC):
      fp.write("\tMFCC_"+str(i))
   for i in range(my_c.numberMFCC):
      fp.write("\tMFCC1_"+str(i))
   for i in range(my_c.nBarkband):
      fp.write("\tBarkband_"+str(i))
   for i in range(my_c.nBarkband):
      fp.write("\tBarkband1_"+str(i))

   fp.write("\n")
   fp.close()


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
#outputFile="OP_Spiel_ich"
outputFile="OP_Ludmila"
outputFile="OP_Shemahanskaya_zarica"

my_c= my_cons()

write_head(my_c.dir_data+outputFile+".dat")
    
harm_anal(my_c.dir_in_sound,my_c.dir_out_sound,outputFile+".flac",my_c.dir_data+outputFile+".dat")
