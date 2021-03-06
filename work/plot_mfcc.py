import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data=pd.read_table("../data/mfcc.dat")

for i_m in range(5,11):
        if i_m==5:
            clr='red'
        elif i_m==6:
            clr='orange'
        elif i_m==7:
            clr='yellow'
        elif i_m==8:
            clr='green'
        elif i_m==9:
            clr='lightblue'
        elif i_m==10:
            clr='blue'
        elif i_m==11:
            clr='violet'
        else:
            clr='black'

        for lbl in ["Arabesque","Bellezza del canto"]:
           sel_data=data[(data.loc[:,"label"]==lbl) & \
             (data.loc[:,"number"]==i_m) ]

           if lbl=="Arabesque":
                ls="-"
           else:
                ls="--"

#           plt.plot(sel_data.loc[:,"frequency"],sel_data.loc[:,"harmonic"],
#              color=clr,linestyle=ls)
           plt.plot(sel_data.loc[:,"n_mfcc"],sel_data.loc[:,"mfcc"], color=clr,linestyle=ls)
plt.xlim((2,12))
plt.show()
