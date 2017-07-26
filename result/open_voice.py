import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.decomposition import PCA

sng_lst= ["OP","AN","VN","LP","MD","JS","AG","AK","LC","AF","IC","KB","ES","RS"]
sng_lst= ["OP","AN","LP","zy"]

data_zy=pd.read_table("zykina_harm.dat")
data_adina=pd.read_table("Adina_harm.dat")
marg=[]
for i in range(12):
   marg.append(500*i)

marg0=[300,600]
nHarm=15
freq_a=[]
for j in range(len(marg)-1):
   freq_a=np.append(freq_a,(marg[j]+marg[j+1])/2.0)

for sng in sng_lst:
  for k in range(len(marg0)-1):
   freq_mid=np.zeros(len(marg)-1)
   harm_mid=np.zeros(len(marg)-1)
   count_mid=np.zeros(len(marg)-1)
   if sng=="zy":
     data_sng=data_zy
   else:
     data_sng=data_adina[data_adina.loc[:,"Singer"]==sng]
   for (idx,row) in data_sng.iterrows():
      fr_b=row.loc["Mean_frequency"]
      if  not marg0[k]<=fr_b<=marg0[k+1]:
          continue
      harm_a=np.empty((len(marg)))
      harm_a[:]=np.nan
      for i in range(1,nHarm):
         if i==1:
            harm=0
         else: 
            harm=row.loc['Harm_p_'+str(i)]
         fr=fr_b*i
         for j in range(len(marg)-1):
            if marg[j]<=fr<=marg[j+1]:
               freq_mid[j]=freq_mid[j]+fr
               harm_mid[j]=harm_mid[j]+harm
               count_mid[j]+=1
               harm_a[j]=harm

      harm_a[len(marg)-1]=k

   for j in range(len(marg)-1):
       if count_mid[j]>0:
           freq_mid[j]=freq_mid[j]/count_mid[j]
           harm_mid[j]=harm_mid[j]/count_mid[j]
       else:
           freq_mid[j]=np.nan
           harm_mid[j]=np.nan
   if sng=="OP":
       col="b"
   elif sng=="AN":
       col="r"
   elif sng=="VN":
       col="g"
   elif sng=="LP":
       col="orange"
   elif sng=="MD":
       col="magenta"
   elif sng=="JS":
       col="cyan"
   elif sng=="AG":
       col="black"
   elif sng=="AK":
       col="yellow"
   elif sng=="LC":
       col="lightblue"
   elif sng=="AF":
       col="lightgreen"
   elif sng=="IC":
       col="g"
   elif sng=="KB":
       col="cyan"
   elif sng=="ES":
       col="b"
   elif sng=="RS":
       col="r"
   elif sng=="zy":
       col="black"

#   plt.plot(freq_a,harm_mid,color=col)
   if k==1:
       col="g"
   plt.plot(freq_mid,harm_mid,color=col)
plt.show()

lst=[]
for k in range(2,16):
   lst.append('Harm_p_'+str(k))
X=data_adina.loc[:,lst]
pca = PCA(n_components=6)
X_c = pca.fit_transform(X)

X_n=pca.inverse_transform(X_c)
fp=open("aaa.dat","w")
mf=data_adina.loc[:,"Mean_frequency"].as_matrix()
sng=data_adina.loc[:,"Singer"].as_matrix()
fp.write("Mean_frequency\tSinger")
for i in range(2,16):
   fp.write("\t"+"Harm_p_"+str(i))
fp.write("\n")
i_rec=0
for rec in X_n:
   fp.write(str(mf[i_rec]))
   fp.write("\t"+sng[i_rec])
   for i in range(14):
      fp.write("\t"+str(rec[i])) 
   fp.write("\n")
   i_rec+=1
fp.close()

data_n=pd.read_table("aaa.dat")

freq_a=[]
for j in range(len(marg)-1):
   freq_a=np.append(freq_a,(marg[j]+marg[j+1])/2.0)

for sng in sng_lst:
  for k in range(len(marg0)-1):
   freq_mid=np.zeros(len(marg)-1)
   harm_mid=np.zeros(len(marg)-1)
   count_mid=np.zeros(len(marg)-1)
   if sng=="zy":
     data_sng=data_zy
   else:
     data_sng=data_n[data_n.loc[:,"Singer"]==sng]
   for (idx,row) in data_sng.iterrows():
      fr_b=row.loc["Mean_frequency"]
      if  not marg0[k]<=fr_b<=marg0[k+1]:
          continue
      harm_a=np.empty((len(marg)))
      harm_a[:]=np.nan
      for i in range(1,nHarm):
         if i==1:
            harm=0
         else: 
            harm=row.loc['Harm_p_'+str(i)]
         fr=fr_b*i
         for j in range(len(marg)-1):
            if marg[j]<=fr<=marg[j+1]:
               freq_mid[j]=freq_mid[j]+fr
               harm_mid[j]=harm_mid[j]+harm
               count_mid[j]+=1
               harm_a[j]=harm

      harm_a[len(marg)-1]=k

   for j in range(len(marg)-1):
       if count_mid[j]>0:
           freq_mid[j]=freq_mid[j]/count_mid[j]
           harm_mid[j]=harm_mid[j]/count_mid[j]
       else:
           freq_mid[j]=np.nan
           harm_mid[j]=np.nan
   if sng=="OP":
       col="b"
   elif sng=="AN":
       col="r"
   elif sng=="VN":
       col="g"
   elif sng=="LP":
       col="orange"
   elif sng=="MD":
       col="magenta"
   elif sng=="JS":
       col="cyan"
   elif sng=="AG":
       col="black"
   elif sng=="AK":
       col="yellow"
   elif sng=="LC":
       col="lightblue"
   elif sng=="AF":
       col="lightgreen"
   elif sng=="IC":
       col="g"
   elif sng=="KB":
       col="cyan"
   elif sng=="ES":
       col="b"
   elif sng=="RS":
       col="r"
   elif sng=="zy":
       col="black"

#   plt.plot(freq_a,harm_mid,color=col)
   if k==1:
       col="g"
   plt.plot(freq_mid,harm_mid,color=col)
plt.show()


