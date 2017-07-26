import essentia as es
from essentia.standard import *
import numpy as np
import matplotlib.pyplot as plt
import os

def prep_data(inputFilename,s_t,e_t):
   print inputFilename
   loader = MonoLoader(filename = inputFilename, sampleRate = sampleRate )
   slicer=Slicer(startTimes=[s_t],endTimes=[e_t],sampleRate=sampleRate)
   audioWriterOut = MonoWriter(filename=inputFilename[0:-4]+"_out.wav")
   audioWriterHarm = MonoWriter(filename=inputFilename[0:-4]+"_harm.wav")
   audioWriterRes = MonoWriter(filename=inputFilename[0:-4]+"_res.wav")

   hma=HprModelAnal(sampleRate=sampleRate,hopSize=hopSize,fftSize=frameSize,nHarmonics=nHarmonics, maxnSines=maxnSines,freqDevOffset=freqDevOffset,freqDevSlope=freqDevSlope,harmDevSlope=harmDevSlope,magnitudeThreshold=magnitudeThreshold,maxFrequency=maxFrequency,maxPeaks=maxPeaks,minFrequency=minFrequency,stocf=stocf) 

   sms=SprModelSynth(sampleRate=sampleRate,hopSize=hopSize,fftSize=frameSize)
   equalLoudness = EqualLoudness()
   predominantMelody=PredominantPitchMelodia(frameSize=frameSize,hopSize=hopSize,sampleRate=sampleRate, filterIterations=3,voicingTolerance=0.2,voiceVibrato=True,minFrequency=80,guessUnvoiced=False)
   window =Windowing(type='hamming')
   spectrum=Spectrum(size=frameSize)
   mfcc=MFCC(numberCoefficients=numberMFCC,inputSize=frameSize/2+1)
   bb=BarkBands(sampleRate=sampleRate)

   audio=slicer(loader())[0]
   eqaudio=equalLoudness(audio)
   pitch,pitchConfidence=predominantMelody(eqaudio)
   sct=SpectralCentroidTime(sampleRate=sampleRate)

   cnt=0
   audioout1=[]
   audioout2=[]
   audioout3=[]
   fp=open(outputFilename,"a")
   k_t=hopSize/float(sampleRate)

   for frame in FrameGenerator(audio, frameSize, hopSize, startFromZero=False):
      if cnt%1000==0:
          print cnt
      frequencies,magnitudes,phases,res=hma(frame,pitch[cnt])
      outframe,harmframe,res=sms(magnitudes,frequencies,phases,res)
      centroid=sct(frame)
      spect=spectrum(window(frame))

#      for ih in range(nHarmonics):
#         if frequencies[ih]==0:
#            frequencies[ih]=frequencies[0]*(ih+1)   
      spect1=np.zeros(spect.shape)
      for i in range(nHarmonics):
         if magnitudes[i]>-100:
            spect1[int(frequencies[i]*frameSize/sampleRate)]=10**(magnitudes[i]/20)
      mfcc_bands,mfcc_coeffs=mfcc(spect)
      mfcc_bands1,mfcc_coeffs1=mfcc(es.array(spect1))
      barkband=bb(spect)  
      barkband1=bb(es.array(spect1))  

      audioout1 = np.append(audioout1,outframe)
      audioout2 = np.append(audioout2,harmframe)
      audioout3 = np.append(audioout3,res)

      fp.write(inputFilename+"\t"+inputFilename[0:2]+"\t"+str(frequencies[0])+"\t"+str(pitch[cnt])+"\t"+str(k_t*cnt))
      for j in range(nHarmonics):
         fp.write("\t"+str(magnitudes[j]))
      for j in range(numberMFCC):
         fp.write("\t"+str(mfcc_coeffs1[j]))
      for j in range(numberMFCC):
         fp.write("\t"+str(mfcc_coeffs[j]))
      for j in range(nBarkband):
         fp.write("\t"+str(barkband[j]))
      for j in range(nBarkband):
         fp.write("\t"+str(barkband1[j]))
      fp.write("\t"+str(centroid))
      fp.write("\n")

      cnt+=1

   fp.close()
   audioWriterOut(audioout1.astype(np.float32))
   audioWriterHarm(audioout2.astype(np.float32))
   audioWriterRes(audioout3.astype(np.float32))


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
numberMFCC=12
nBarkband=27
outputFilename="Peretyatko.dat"

if os.path.exists(outputFilename):
    os.remove(outputFilename)
    
fp=open(outputFilename,"w")

fp.write("Filename\tSinger\tFrequency\tPitch\tTime")
for i in range(nHarmonics):
   fp.write("\tHarm_"+str(i))
for i in range(numberMFCC):
   fp.write("\tMFCC_"+str(i))
for i in range(numberMFCC):
   fp.write("\tMFCC1_"+str(i))
for i in range(nBarkband):
   fp.write("\tBarkBand_"+str(i))
for i in range(nBarkband):
   fp.write("\tBarkBand1_"+str(i))
fp.write("\tCentroid")
fp.write("\n")
fp.close()


#prep_data("OP_bacio.wav",0,31)
#prep_data('OP_Adina.aac',117,205)
#prep_data('AN_Adina.wav',97,192)
#prep_data('VN_Adina.aac',105,193)
#prep_data('LP_Adina.wav',0,82)
#prep_data('MD_Adina.wav',0,93)
#prep_data('JS_Adina.wav',0,80)
#prep_data('AG_Adina.wav',0,82)
#prep_data('AK_Adina.wav',0,82)
#prep_data('LC_Adina.wav',0,71)
#prep_data('AF_Adina.wav',0,81)
#prep_data('IC_Adina.wav',0,85)
#prep_data('KB_Adina.wav',0,86)
#prep_data('ES_Adina.wav',0,76)
#prep_data('RS_Adina.wav',0,89)
#prep_data('OP_strano.wav',0,82)
#prep_data('VG_strano.wav',0,78)
#prep_data('AN_Lauretta.wav',0,170)

prep_data("OP_SeInCiela.wav",0,25)
prep_data("OP_NonMiDira.wav",0,26)
#prep_data("OP_bacio.wav",0,31)
prep_data("OP_Bacio1a.wav",0,32)
prep_data("OP_Bacio2a.wav",0,27)
prep_data("OP_Bacio3a.wav",0,31)
