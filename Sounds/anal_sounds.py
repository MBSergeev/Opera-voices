import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import smooth
import sys

from sklearn.linear_model import LogisticRegression


data=pd.read_table("Voices.dat")

data=data.assign(Harm2=lambda x: x.nHarm_2-x.nHarm_1)
data=data.assign(Harm3=lambda x: x.nHarm_3-x.nHarm_1)
data=data.assign(Harm4=lambda x: x.nHarm_4-x.nHarm_1)
data=data.assign(Harm5=lambda x: x.nHarm_5-x.nHarm_1)
data=data.assign(Harm6=lambda x: x.nHarm_6-x.nHarm_1)


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

data_s1au=data_s1.loc[(data_s1.sound=="a") | (data_s1.sound=="u")]



X=data_s1au[["fr0_mean","Harm2","Harm4"]].to_numpy()

y=data_s1au["sound"].to_numpy()

y=(y=="a").astype(int)

clf = LogisticRegression(random_state=0).fit(X, y)

y_pr=clf.predict(X)

print(y)
print(y_pr)
