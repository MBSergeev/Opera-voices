import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def read_data(inputFile):

  fp=open(inputFile,"r")
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
  return (data,flds)

np.set_printoptions(threshold='nan')

data,flds=read_data("Adina.dat")



marg=[0,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000]
#marg=[700,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000]
#marg0=[300,400,500,600,700,800,900,1000]
marg0=[300,1000]
nHarm=16

freq_a=[]
for j in range(len(marg)-1):
   freq_a=np.append(freq_a,(marg[j]+marg[j+1])/2.0)
# data_n - harmonics in ranges
#data_n=np.empty((0,len(marg)-1))
data_n=np.empty((0,len(marg)))
sng_n=[]

data_mid=np.empty((0,len(marg)-1))
sng_lst= ["OP","AN","VN","LP","MD","JS","AG","AK","LC","AF","IC","KB","ES","RS"]
#sng_lst= ["OP","AN","AK","JS"]#,"AN","VN","LP","MD","JS","AG","AK","LC","AF","IC","KB","ES","RS"]
for sng in sng_lst:
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

  for k in range(len(marg0)-1):
#   sng="MC"
#   data_sng=data
   data_sng=data[np.where(data[:,flds['Singer']]==sng)]

   freq_mid=np.zeros(len(marg)-1)
   harm_mid=np.zeros(len(marg)-1)
   count_mid=np.zeros(len(marg)-1)
   for rec in data_sng:
      fr_b=np.float64(rec[flds['Mean_frequency']])
      if  not marg0[k]<=fr_b<=marg0[k+1]:
          continue
      harm_a=np.empty((len(marg)-1))
      harm_a=np.empty((len(marg)))
      harm_a[:]=np.nan
      for i in range(1,nHarm):
         if i==1:
            harm=0
         else: 
            harm=np.float64(rec[flds['Harm_p_'+str(i)]])
#            harm=np.float64(rec[flds['Harm_'+str(i)]])
         fr=fr_b*i
         for j in range(len(marg)-1):
            if marg[j]<=fr<=marg[j+1]:
               freq_mid[j]=freq_mid[j]+fr
               harm_mid[j]=harm_mid[j]+harm
               count_mid[j]+=1
               harm_a[j]=harm
      harm_a[len(marg)-1]=k
      data_n=np.append(data_n,[harm_a],axis=0)
      sng_n=np.append(sng_n,sng)
#      plt.scatter(freq_a,harm_a[0:-1],color=col)
   for j in range(len(marg)-1):
       if count_mid[j]>0:
           freq_mid[j]=freq_mid[j]/count_mid[j]
           harm_mid[j]=harm_mid[j]/count_mid[j]
       else:
           freq_mid[j]=np.nan
           harm_mid[j]=np.nan

#   plt.plot(freq_mid,harm_mid,color='b')
   plt.plot(freq_a,harm_mid,color=col)
   data_mid=np.append(data_mid,[harm_mid],axis=0)


plt.show()

data_mid[np.isnan(data_mid)]=0

matr=np.zeros((len(sng_lst),len(sng_lst)))
for i in range(len(sng_lst)):
   for j in range(len(sng_lst)):
      sss=0
      for k in range(1,data_mid.shape[1]):
        sss+=(data_mid[i,k]-data_mid[j,k])**2
      matr[i,j]=np.sqrt(sss)

for i in range(len(sng_lst)):
  ddd=matr[i]
  print sng_lst[i]
  eee1=np.argsort(ddd)
  eee2=np.sort(ddd)
  for j in range(len(eee1)):
     print sng_lst[eee1[j]],eee2[j]
   


col_mean=stats.nanmean(data_n,axis=0)
inds=np.where(np.isnan(data_n))
data_n[inds]=np.take(col_mean,inds[1])


X=data_n
y=sng_n

X = X[:,:len(marg)-1]
#print X

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
n_f=4
kf=KFold(n_splits=n_f,shuffle=True,random_state=42)
s_score=0

for train_index,test_index in kf.split(X):
   X_train,X_test =X[train_index],X[test_index]
   y_train,y_test =y[train_index],y[test_index]
   
   neigh=KNeighborsClassifier(n_neighbors=3)
   neigh.fit(X_train,y_train)
   sc = neigh.score(X_test,y_test)
   print "score=",sc
   s_score+=sc

print "Mean score=",s_score/n_f

neigh=KNeighborsClassifier(n_neighbors=3)
neigh.fit(X,y)
print neigh.score(X,y)
"""
lst=[flds['Mean_frequency'],flds['Harm_p_2'],flds['Harm_p_3'],flds['Harm_p_4'],flds['Harm_p_5'],\
    flds['Harm_p_6'],flds['Harm_p_7'],flds['Harm_p_8'],flds['Harm_p_9'],\
    flds['Harm_p_10'],flds['Harm_p_11'],flds['Harm_p_12'],\
    flds['Harm_p_13'],flds['Harm_p_14'],flds['Harm_p_15']]


X=data[:,lst]
X=np.float64(X)
y=data[:,flds['Singer']]
fltr=~np.isnan(X).any(axis=1)
X=X[fltr]
y=y[fltr]

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
n_f=4
kf=KFold(n_splits=n_f,shuffle=True,random_state=42)
s_score=0

for train_index,test_index in kf.split(X):
   X_train,X_test =X[train_index],X[test_index]
   y_train,y_test =y[train_index],y[test_index]

   neigh=KNeighborsClassifier(n_neighbors=3)
   neigh.fit(X_train,y_train)
   sc = neigh.score(X_test,y_test)
   print "score=",sc
   s_score+=sc

print "Mean score=",s_score/n_f

neigh=KNeighborsClassifier(n_neighbors=3)
neigh.fit(X,y)
print neigh.score(X,y)
"""

from sklearn.cluster import KMeans

n_clusters=2
kmeans=KMeans(n_clusters=n_clusters,random_state=42)
kmeans.fit(X,y)
pred=kmeans.predict(X)


for sng in sng_lst:
    s=np.zeros(n_clusters)
    for rec in zip(pred,y):
      if rec[1]==sng:
         s[rec[0]]+=1
    s1=s/sum(s)*100
    
#    print sng,np.argmax(s1),max(s1),s
   

ind=np.where(sng_n=="OP")
for sng in ["VN","JS","KB"]:
   ind=np.append(ind,np.where(sng_n==sng))
X1=X[ind]
y1=y[ind]
n_clusters=8
kmeans=KMeans(n_clusters=n_clusters,random_state=42)
kmeans.fit(X,y)
pred=kmeans.predict(X)
print "Group 0"
for sng in ["OP","VN","JS","KB"]:
    s=np.zeros(n_clusters)
    for rec in zip(pred,y):
      if rec[1]==sng:
         s[rec[0]]+=1
    s1=s/sum(s)*100

#    print sng,np.argmax(s1),max(s1),s

ind=np.where(sng_n=="AN")
for sng in ["LP","MD","AG","AK","LC","AF","IC","ES","RS"]:
   ind=np.append(ind,np.where(sng_n==sng))
X1=X[ind]
y1=y[ind]
n_clusters=10
kmeans=KMeans(n_clusters=n_clusters,random_state=42)
kmeans.fit(X,y)
pred=kmeans.predict(X)
print "Group 1"
for sng in ["AN","LP","MD","AG","AK","LC","AF","IC","ES","RS"]:
    s=np.zeros(n_clusters)
    for rec in zip(pred,y):
      if rec[1]==sng:
         s[rec[0]]+=1
    s1=s/sum(s)*100

#    print sng,np.argmax(s1),max(s1),s

