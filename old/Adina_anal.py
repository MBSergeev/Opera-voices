import numpy as np
import matplotlib.pyplot as plt

fp=open("Adina.dat","r")
sss=fp.readline()
cnt=0
while sss!="":
   sss=sss[0:-1]
   rec=sss.split("\t")
   if cnt==0:
       flds={} 	
       i=0
       for f in rec:
          flds[f]=i
          i+=1
       data=np.empty((0,len(flds)))

   else:
       rec_n=[]
       for r in rec:
          rec_n=np.append(rec_n,r)
        
       data=np.append(data,[rec_n],axis=0)

       
     
   cnt+=1  
   sss=fp.readline()
   
   
fp.close()
np.set_printoptions(threshold='nan')


OP=data[np.where(data[:,flds['Singer']]=='OP')]
AN=data[np.where(data[:,flds['Singer']]=='AN')]
VN=data[np.where(data[:,flds['Singer']]=='VN')]
LP=data[np.where(data[:,flds['Singer']]=='LP')]
MD=data[np.where(data[:,flds['Singer']]=='MD')]

"""
plt.scatter(OP[:,flds['Harm_2']],OP[:,flds['Harm_p_2']],c='b')
plt.scatter(AN[:,flds['Harm_2']],AN[:,flds['Harm_p_2']],c='r')
plt.scatter(VN[:,flds['Harm_2']],VN[:,flds['Harm_p_2']],c='g')
plt.scatter(LP[:,flds['Harm_2']],LP[:,flds['Harm_p_2']],c='orange')
plt.scatter(MD[:,flds['Harm_2']],MD[:,flds['Harm_p_2']],c='magenta')
plt.scatter(JS[:,flds['Harm_2']],JS[:,flds['Harm_p_2']],c='cyan')


plt.show()
"""
from sklearn import linear_model
regr = linear_model.LinearRegression()
h2=np.float64(OP[:,flds['Harm_2']].reshape(-1,1))
hp2=np.float64(OP[:,flds['Harm_p_2']])
"""
regr.fit(h2,hp2)
print regr.coef_
print np.transpose(np.vstack((regr.predict(h2),hp2,hp2-regr.predict(h2))))
print OP[:,flds['Filename']]
print OP[:,flds['Harm_p_2']]
"""
lst=[flds['Harm_p_2'],flds['Harm_p_3'],flds['Harm_p_5'],flds['Harm_p_10']]
"""
lst=[flds['Harm_p_2'],flds['Harm_p_3'],flds['Harm_p_4'],flds['Harm_p_5'],\
    flds['Harm_p_6'],flds['Harm_p_7'],flds['Harm_p_8'],flds['Harm_p_9'],\
    flds['Harm_p_10'],flds['Harm_p_11'],flds['Harm_p_12'],\
    flds['Harm_p_13'],flds['Harm_p_14'],flds['Harm_p_15']]
lst=[flds['Harm_2'],flds['Harm_3'],flds['Harm_4'],flds['Harm_5'],\
    flds['Harm_6'],flds['Harm_p_7'],flds['Harm_8'],flds['Harm_9'],\
    flds['Harm_10'],flds['Harm_11'],flds['Harm_12'],\
    flds['Harm_13'],flds['Harm_14'],flds['Harm_15']]
"""
lst=[flds['Mean_frequency'],flds['Harm_p_2'],flds['Harm_p_3'],flds['Harm_p_4'],flds['Harm_p_5'],\
    flds['Harm_p_6'],flds['Harm_p_7'],flds['Harm_p_8'],flds['Harm_p_9'],\
    flds['Harm_p_10'],flds['Harm_p_11'],flds['Harm_p_12'],\
    flds['Harm_p_13'],flds['Harm_p_14'],flds['Harm_p_15']]
X=data[:,lst]
X=np.float64(X)
#y=np.where(data[:,flds['Singer']]=='OP',2,1)
y=data[:,flds['Singer']]


#fltr=np.where(data[:,flds['Singer']]=='AN',False,True)
#X=X[fltr]
#y=y[fltr]

#fltr=~np.isnan(X).any(axis=1)
#X=X[fltr]
#y=y[fltr]

from sklearn.neighbors import KNeighborsClassifier
neigh=KNeighborsClassifier(n_neighbors=3)
neigh.fit(X,y)
print neigh.score(X,y)
"""
hs=np.empty((0,15))
for sng in ["OP","AN","VN","LP","MD","JS"]:
    sel_data=data[np.where(data[:,flds['Singer']]==sng)]
    print sng,np.mean(np.float64(sel_data[:,flds['Mean_frequency']]))
    lst=[sng,\
    np.mean(np.float64(sel_data[:,flds['Harm_p_2']])),\
    np.mean(np.float64(sel_data[:,flds['Harm_p_3']])),\
    np.mean(np.float64(sel_data[:,flds['Harm_p_4']])),\
    np.mean(np.float64(sel_data[:,flds['Harm_p_5']])),\
    np.mean(np.float64(sel_data[:,flds['Harm_p_6']])),\
    np.mean(np.float64(sel_data[:,flds['Harm_p_7']])),\
    np.mean(np.float64(sel_data[:,flds['Harm_p_8']])),\
    np.mean(np.float64(sel_data[:,flds['Harm_p_9']])),\
    np.mean(np.float64(sel_data[:,flds['Harm_p_10']])),\
    np.mean(np.float64(sel_data[:,flds['Harm_p_11']])),\
    np.mean(np.float64(sel_data[:,flds['Harm_p_12']])),\
    np.mean(np.float64(sel_data[:,flds['Harm_p_13']])),\
    np.mean(np.float64(sel_data[:,flds['Harm_p_14']])),\
    np.mean(np.float64(sel_data[:,flds['Harm_p_15']]))\
    ]
    ""
    lst=[sng,\
    np.mean(np.float64(sel_data[:,flds['Harm_2']])),\
    np.mean(np.float64(sel_data[:,flds['Harm_3']])),\
    np.mean(np.float64(sel_data[:,flds['Harm_4']])),\
    np.mean(np.float64(sel_data[:,flds['Harm_5']])),\
    np.mean(np.float64(sel_data[:,flds['Harm_6']])),\
    np.mean(np.float64(sel_data[:,flds['Harm_7']])),\
    np.mean(np.float64(sel_data[:,flds['Harm_8']])),\
    np.mean(np.float64(sel_data[:,flds['Harm_9']])),\
    np.mean(np.float64(sel_data[:,flds['Harm_10']])),\
    np.mean(np.float64(sel_data[:,flds['Harm_11']])),\
    np.mean(np.float64(sel_data[:,flds['Harm_12']])),\
    np.mean(np.float64(sel_data[:,flds['Harm_13']])),\
    np.mean(np.float64(sel_data[:,flds['Harm_14']])),\
    np.mean(np.float64(sel_data[:,flds['Harm_15']]))\
    ]
    "" 
    
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
    plt.plot(lst[1:],color=col)
    hs=np.append(hs,[lst],axis=0)

#plt.show()
#print hs
"""
"""
for rec in data:
   sng=rec[flds['Singer']]
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
   freq=[]
   harm=[]
   if sng!="LP" and sng!="MD":
      continue
   print rec[flds['Filename']]
   fr=float(rec[flds['Mean_frequency']])
   if not (700<=fr<=900):
      continue
   for i in range(1,16):
      freq.append(fr*i)
      if i==1:
         harm.append(0.0)
      else:
         harm.append(float(rec[flds['Harm_p_'+str(i)]]))
   plt.plot(freq,harm,color=col)

plt.show()
"""    
#mf= data[:,flds['Mean_frequency']]
#print np.min(np.float64(mf))
#print np.max(np.float64(mf))

marg=[0,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,6000]
marg0=[300,1000]

for sng in ["OP","AN","VN","LP","MD","JS","AG","AK","LC","AF","IC","KB","ES","RS"]:
  for k in range(len(marg0)-1):
#   sng="MC"
   data_sng=data
   data_sng=data[np.where(data[:,flds['Singer']]==sng)]

   freq_mid=np.zeros(len(marg)-1)
   harm_mid=np.zeros(len(marg)-1)
   count_mid=np.zeros(len(marg)-1)
   for rec in data_sng:
      fr_b=np.float64(rec[flds['Mean_frequency']])
      if  not marg0[k]<=fr_b<=marg0[k+1]:
          continue
   
      for i in range(1,16):
         if i==1:
            harm=0
         else: 
            harm=np.float64(rec[flds['Harm_p_'+str(i)]])
         fr=fr_b*i
         for j in range(len(marg)-1):
            if marg[j]<=fr<=marg[j+1]:
               freq_mid[j]=freq_mid[j]+fr
               harm_mid[j]=harm_mid[j]+harm
               count_mid[j]+=1
          
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
       col="black"

   plt.plot(freq_mid,harm_mid,color=col)

plt.show()
