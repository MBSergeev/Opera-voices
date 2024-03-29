import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data=pd.read_table("formants.dat").assign(f2_f1=lambda x: x.f2-x.f1)

data_a=data[data.sound=="a"]
data_o=data[data.sound=="o"]
data_e=data[data.sound=="e"]
data_u=data[data.sound=="u"]
data_i=data[data.sound=="i"]

plt.scatter(data_a["f2_f1"],data_a["f1"],color="red")
plt.scatter(data_o["f2_f1"],data_o["f1"],color="orange")
plt.scatter(data_e["f2_f1"],data_e["f1"],color="green")
plt.scatter(data_u["f2_f1"],data_u["f1"],color="blue")
plt.scatter(data_i["f2_f1"],data_i["f1"],color="black")

plt.show()



X=data[["f2_f1","f1"]].to_numpy()

y=data["sound"].to_numpy()


clf = LogisticRegression(random_state=0,max_iter=1000,multi_class="multinomial").fit(X, y)

y_pr=clf.predict(X)
print(y)
print(y_pr)

print(accuracy_score(y,y_pr))