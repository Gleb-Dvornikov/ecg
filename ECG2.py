import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

column_names=['t','ecg']

ecg=pd.read_csv('ЭКГ.txt', names=column_names,
                      sep='\t',skiprows=0,skipfooter=0,engine='python' )

ecg.t=ecg.t/1000

print(ecg)
plt.plot(ecg.t,ecg.ecg)
plt.xlabel("t")
plt.ylabel("ecg")
plt.show()