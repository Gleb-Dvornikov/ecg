import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import signal

data = np.loadtxt('ЭКГ.txt', delimiter='\t', dtype=np.float64)
data = data[1031:1313] #174:574; 574:1031; 1031:1313
filter = signal.firwin(10, [2, 70], fs=250, pass_zero=False)
filtered = signal.lfilter(filter, 1.0, data[:, 0])
trr = data[:, 1][data[:, 1] > 300]
trr = data[:, 1][data[:, 1] < 1400]
M, O = np.mean(trr), np.std(trr)
index = 0
NN = []
timeNN = []
for rr in trr:
    if (rr < (M + 3 * O)) and (rr > (M - 3 * O)):
        NN.append(rr)
        if index == 0:
            timeNN.append(0)
        else:
            timeNN.append(timeNN[index - 1] + NN[index - 1])
        index = index + 1
timeNN = np.array(timeNN, dtype=int)
NN = np.array(NN)
n = max([np.round((max(NN) - min(NN)) / 50), 1])
n = np.array(n, dtype=int)
am, m = np.histogram(trr, int(n))
am = 100 * am / sum(am)
am = np.insert(am, 0, 0)
AM0 = max(am)
Nmax = np.argmax(am)
M0 = m[Nmax]
Y = m[am > 3]
VR = max([Y[-1] - Y[0], 50.0])
plt.hist(trr, n)
label_font = {'fontname': 'Times New Roman', 'size': '25'}
plt.xlabel('R-R интервалы, мс', **label_font)
plt.ylabel('количество R-R в бине, шт', **label_font)
plt.show()
print(scipy.stats.kstest(NN, 'norm'))
print(M0)
print(VR)
print(AM0)

print("ИН ", AM0/(2*M0*VR)*10**6)
print("ИВР ", AM0/VR*1000)
print("ВПР ", 1/(M0*VR)*10**6)
print("ПАПР ", AM0/M0*1000)
