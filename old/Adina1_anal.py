import numpy as np
import matplotlib.pyplot as plt

fp=open("Adina1.dat","r")
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
JS=data[np.where(data[:,flds['Singer']]=='JS')]

#plt.scatter(OP[:,flds['Harm_p_2']],OP[:,flds['Harm_p_5']],c='b')
#plt.scatter(AN[:,flds['Harm_p_2']],AN[:,flds['Harm_p_5']],c='r')
#plt.scatter(VN[:,flds['Harm_p_2']],VN[:,flds['Harm_p_2']],c='g')
#plt.scatter(LP[:,flds['Harm_p_2']],LP[:,flds['Harm_p_5']],c='orange')
#plt.scatter(MD[:,flds['Harm_p_2']],MD[:,flds['Harm_p_5']],c='magenta')
#plt.scatter(JS[:,flds['Harm_p_2']],JS[:,flds['Harm_p_5']],c='cyan')
#plt.show()

lst=[flds['Mean_frequency'],flds['Harm_p_2'],flds['Harm_p_3'],flds['Harm_p_4'],flds['Harm_p_5']]
lst=[flds['Harm_p_2'],flds['Harm_p_3'],flds['Harm_p_5'],flds['Harm_p_10'],flds['Harm_p_13'],flds['Harm_p_14']]

X=data[:,lst]
X=np.float64(X)
i#y=np.where(data[:,flds['Singer']]=='OP',2,1)
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

hs=np.empty((0,7))
for sng in ["OP","AN","VN","LP","MD","JS"]:
    sel_data=data[np.where(data[:,flds['Singer']]==sng)]
#    np.mean(np.float64(sel_data[:,flds['Mean_frequency']]))
    lst=[sng,\
    np.mean(np.float64(sel_data[:,flds['Harm_p_2']])),\
    np.mean(np.float64(sel_data[:,flds['Harm_p_3']])),\
    np.mean(np.float64(sel_data[:,flds['Harm_p_5']])),\
    np.mean(np.float64(sel_data[:,flds['Harm_p_10']])),\
    np.mean(np.float64(sel_data[:,flds['Harm_p_13']])),\
    np.mean(np.float64(sel_data[:,flds['Harm_p_14']]))\
    ]
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

plt.show()
print hs
