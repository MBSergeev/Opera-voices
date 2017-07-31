class my_cons:

  def __init__(self):
    self.frameSize = 2*1024
    self.hopSize = 256
    self.sampleRate = 44100
    self.nHarmonics=15
    self.freqDevOffset=20
    self.freqDevSlope=0.01
    self.harmDevSlope=0.01
    self.magnitudeThreshold =0
    self.maxFrequency=15000
    self.maxPeaks=100
    self.maxnSines=100
    self.minFrequency=20
    self.stocf = 0.2
    self.numberMFCC=16
    self.nBarkband=27

    self.dir_in_sound="../in_sound/"
    self.dir_out_sound="../out_sound/"
    self.dir_data="../data/"

    self.marg=[0,50,100,150,200,300,400,510,630,770,920,1080,1270,1480,1720,2000,2320,2700,3150,3700,4400,5300,6400,7700,9500,12000,15500,20500]#,27000]

