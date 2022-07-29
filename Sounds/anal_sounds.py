from statistics import harmonic_mean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import smooth
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data=pd.read_table("Voices.dat")

data=data.assign(Harm2=lambda x: x.nHarm_2-x.nHarm_1)
data=data.assign(Harm3=lambda x: x.nHarm_3-x.nHarm_1)
data=data.assign(Harm4=lambda x: x.nHarm_4-x.nHarm_1)
data=data.assign(Harm5=lambda x: x.nHarm_5-x.nHarm_1)
data=data.assign(Harm6=lambda x: x.nHarm_6-x.nHarm_1)
data=data.assign(Harm7=lambda x: x.nHarm_7-x.nHarm_1)
data=data.assign(Harm8=lambda x: x.nHarm_8-x.nHarm_1)
data=data.assign(Harm9=lambda x: x.nHarm_9-x.nHarm_1)
data=data.assign(Harm10=lambda x: x.nHarm_10-x.nHarm_1)
data=data.assign(Harm11=lambda x: x.nHarm_11-x.nHarm_1)
data=data.assign(Harm12=lambda x: x.nHarm_12-x.nHarm_1)
data=data.assign(Harm13=lambda x: x.nHarm_13-x.nHarm_1)
data=data.assign(Harm14=lambda x: x.nHarm_14-x.nHarm_1)
data=data.assign(Harm15=lambda x: x.nHarm_15-x.nHarm_1)



data_s1=data[data.voice=="Soprano1"]
data_s2=data[data.voice=="Soprano2"]
data_s3=data[data.voice=="Soprano3"]
data_m1=data[data.voice=="Mezzo1"]
data_m2=data[data.voice=="Mezzo2"]
data_m3=data[data.voice=="Mezzo3"]
data_m4=data[data.voice=="Mezzo4"]
data_t1=data[data.voice=="Tenor1"]
data_t2=data[data.voice=="Tenor2"]
data_b1=data[data.voice=="Baritone1"]
data_b2=data[data.voice=="Baritone2"]
"""

# зависимость частоты от ноты

plt.scatter(data_s1["n"],data_s1["fr0_mean"],color="r")
plt.scatter(data_s2["n"],data_s2["fr0_mean"],color="orange")
plt.scatter(data_s3["n"],data_s3["fr0_mean"],color="yellow")

plt.scatter(data_m1["n"],data_m1["fr0_mean"],color="green")
plt.scatter(data_m2["n"],data_m2["fr0_mean"],color="lightblue")
plt.scatter(data_m3["n"],data_m3["fr0_mean"],color="blue")
plt.scatter(data_m4["n"],data_m4["fr0_mean"],color="magenta")

plt.show()

plt.scatter(data_t1["n"],data_t1["fr0_mean"],color="lightblue")
plt.scatter(data_t2["n"],data_t2["fr0_mean"],color="blue")
plt.scatter(data_b1["n"],data_b1["fr0_mean"],color="magenta")
plt.scatter(data_b2["n"],data_b2["fr0_mean"],color="black")
plt.show()
"""

"""
# зависимость гармоник для разных гласных

data_s1a=data_s1[data_s1.sound=="a"]
data_s1u=data_s1[data_s1.sound=="u"]
data_s1o=data_s1[data_s1.sound=="o"]
data_s1e=data_s1[data_s1.sound=="e"]
data_s1i=data_s1[data_s1.sound=="i"]

plt.scatter(data_s1a["Harm2"],data_s1a["Harm4"],color="red")
plt.scatter(data_s1u["Harm2"],data_s1u["Harm4"],color="black")
plt.scatter(data_s1i["Harm2"],data_s1i["Harm4"],color="green")
plt.scatter(data_s1o["Harm2"],data_s1o["Harm4"],color="blue")
plt.scatter(data_s1e["Harm2"],data_s1e["Harm4"],color="orange")

plt.show()

"""
"""
# логистическая регрессия - гласная от гармоник и частоты

data_s1au=data_s1.loc[(data_s1.sound=="a") | (data_s1.sound=="u")]



X=data_s1au[["fr0_mean","Harm2","Harm4"]].to_numpy()

y=data_s1au["sound"].to_numpy()

y=(y=="a").astype(int)

clf = LogisticRegression(random_state=0).fit(X, y)

y_pr=clf.predict(X)

print(y)
print(y_pr)

"""

"""
# спектр одной ноты для различных гласных

for v in ["a","o","e","u","i"]:
    data_n=data[(data.voice=="Soprano1") & (data.sound==v) & (data.n==10)]

    harm=[]
    harm=np.append(harm,0.0)
    harm=np.append(harm,data_n.Harm2)
    harm=np.append(harm,data_n.Harm3)
    harm=np.append(harm,data_n.Harm4)
    harm=np.append(harm,data_n.Harm5)
    harm=np.append(harm,data_n.Harm6)
    harm=np.append(harm,data_n.Harm7)
    harm=np.append(harm,data_n.Harm8)
    harm=np.append(harm,data_n.Harm9)
    harm=np.append(harm,data_n.Harm10)
    harm=np.append(harm,data_n.Harm11)
    harm=np.append(harm,data_n.Harm12)
    harm=np.append(harm,data_n.Harm13)
    harm=np.append(harm,data_n.Harm14)
    harm=np.append(harm,data_n.Harm15)

    fr=np.arange(1,11)*data_n.fr0_mean.to_numpy().astype(float)

    if v=="a":
        col="red"
    elif v=="o":
        col="orange"
    elif v=="e":
        col="yellow"        
    elif v=="u":
        col="green"
    elif v=="i":
        col="lightblue"   

    plt.plot(fr,harm,color=col)
plt.show()

"""


data_n=data_m4.assign(log_fr=lambda x: np.log(x.fr0_mean))

#lst=["log_fr"]
lst=["fr0_mean"]
#lst=[]
for i in range(2,16):
    lst.append("Harm"+str(i))

X=data_n[lst].to_numpy()

y=data_n["sound"].to_numpy()

#y=(y=="a").astype(int)

clf = LogisticRegression(random_state=0,max_iter=1000,multi_class="auto").fit(X, y)

y_pr=clf.predict(X)
print(y)
print(y_pr)

print(accuracy_score(y,y_pr))

