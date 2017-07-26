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

data,flds=read_data("Adina_mfcc.dat")
#data=data[np.where(data[:,flds['Singer']]!="MC")]
sng_lst= ["OP","AN","VN","LP","MD","JS","AG","AK","LC","AF","IC","KB","ES","RS","MC"]
#sng_lst= ["OP","LP"]


for sng in sng_lst:
   data_sng=data[np.where(data[:,flds['Singer']]==sng)]
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
#   plt.scatter(data_sng[:,flds['MFCC1_2']],data_sng[:,flds['MFCC1_3']],c=col)
   
#plt.show()

#fltr=~np.isnan(X).any(axis=1)
#X=X[fltr]
#y=y[fltr]

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
n_f=6
kf=KFold(n_splits=n_f,shuffle=True,random_state=42)

lst0=[]
for k in range(2,16):
   lst0.append(flds['MFCC1_'+str(k)])

data_n=np.empty((0,len(flds)))
for rec_n in data:
     if rec_n[flds['Singer']] in sng_lst:
        data_n=np.append(data_n,[rec_n],axis=0)


lst=[]
#for n in range(0,11):
#for n in range(7):
n_fact=11
max_res=0
max_n_bin=""
for n in range(1,2**n_fact):
   lst=[]
   n_bin=bin(n)[2:] 
   n_bin='0'*((n_fact)-len(n_bin))+n_bin
   n_bin=n_bin[::-1]
   for jjj in range(len(n_bin)):
      if n_bin[jjj]=='1':
         lst.append(lst0[jjj])
   print "n_bin",n_bin
   

   X=data_n[:,lst]
   X=np.float64(X)
   y=data_n[:,flds['Singer']]

   m_score_max=0
   for n_b in range(2,20):

      s_score=0
      for train_index,test_index in kf.split(X):
         X_train,X_test =X[train_index],X[test_index]
         y_train,y_test =y[train_index],y[test_index]
   
         neigh=KNeighborsClassifier(n_neighbors=n_b)
         neigh.fit(X_train,y_train)
         sc = neigh.score(X_test,y_test)
#      print "score=",sc
         s_score+=sc

#      print "n_neigbors=",n_b,"Mean score=",s_score/n_f
      if s_score/n_f>m_score_max:
          m_score_max=s_score/n_f
          n_b_max=n_b
   print  n,"n_b_max=",n_b_max,"m_score_max=",m_score_max
   if m_score_max>max_res:
       max_res=m_score_max
       max_n_bin=n_bin

print max_res,max_n_bin

from sklearn.cluster import KMeans

n_clusters=3
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


for sng1 in sng_lst:
    d1=np.mean(X[np.where(y==sng1)],axis=0)
    d_lst=[]
    for sng2 in sng_lst:
       d2=np.mean(X[np.where(y==sng2)],axis=0)
       d_lst=np.append(d_lst,np.sqrt(np.sum((d2-d1)**2)))
    l1=np.sort(d_lst)
    l2=np.argsort(d_lst)
    for i in range(len(d_lst)):
#        print sng1,sng_lst[l2[i]],l1[i]
        pass
for i in range(5):    
  for sng in sng_lst[0:15]:
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
     elif sng=="MC":
       col="black"

     s_data=data[np.where(data[:,flds['Singer']]==sng)]
#     plt.scatter(np.mean(np.float64(s_data[:,flds['MFCC1_2']])),np.mean(np.float64(s_data[:,flds['MFCC1_'+str(i+3)]])),color=col)
#     plt.text(np.mean(np.float64(s_data[:,flds['MFCC1_2']])),np.mean(np.float64(s_data[:,flds['MFCC1_'+str(i+3)]])),sng)
#  plt.show()


