import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal


column_names=['t','ecg']

ecg=pd.read_csv('ЭКГ.txt', names=column_names,
                      sep='\t',engine='python' )

ecg.t=ecg.t/60000
print(ecg)
plt.plot(ecg.t,ecg.ecg)
plt.title('Глеб Дворников ВСР')
plt.xlabel("t,минуты")
plt.ylabel("ЭКГ- I:RR")
plt.grid()
plt.show()