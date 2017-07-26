import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier


data=pd.read_table("Adina_harm.dat")
lst=[]
for k in range(2,16):
   lst.append('Harm_p_'+str(k))

X=data.loc[:,lst]
y=data.loc[:,'Singer']


n_f=4
kf=KFold(n_splits=n_f,shuffle=True,random_state=242)
#for n_c in range(1,15):
for n_c in range(6,7):
  pca = PCA(n_components=n_c)
  m_score_max=0
  for n_b in range(2,20):
    X_c = pca.fit_transform(X)

    s_score=0
    for train_index,test_index in kf.split(X):
        X_train,X_test =X_c[train_index],X_c[test_index]
        y_train,y_test =y.iloc[train_index],y.iloc[test_index]

        neigh=KNeighborsClassifier(n_neighbors=n_b)
        neigh.fit(X_train,y_train)
        sc = neigh.score(X_test,y_test)
#      print "score=",sc
        s_score+=sc

#  print "n_neigbors=",n_b,"Mean score=",s_score/n_f
    if s_score/n_f>m_score_max:
            m_score_max=s_score/n_f
            n_b_max=n_b
  print  "n_comp=",n_c,"n_b_max=",n_b_max,"m_score_max=",m_score_max

X_n=pca.inverse_transform(X_c)
#for i in range(200,205):
#   plt.plot( X.as_matrix()[i],'b')
#   plt.plot(X_n[i],'r')
#   plt.show()
from sklearn.cluster import KMeans

sng_lst= ["OP","AN","VN","LP","MD","JS","AG","AK","LC","AF","IC","KB","ES","RS"]


n_clusters=4
kmeans=KMeans(n_clusters=n_clusters,random_state=42)
y=y.as_matrix()
kmeans.fit(X_n,y)
pred=kmeans.predict(X)

sum1=0
sum2=0
for sng in sng_lst:
    s=np.zeros(n_clusters)
    for rec in zip(pred,y):
      if rec[1]==sng:
         s[rec[0]]+=1
    s1=s/sum(s)*100
    
    print sng,np.argmax(s1),max(s1),s
    sum1+=max(s)
    sum2+=sum(s)
print sum1,sum2,float(sum1)*100/sum2
   
sys.exit(0)

