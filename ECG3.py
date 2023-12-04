import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy import signal


def get_interpolation(time, signal, D):
    start = time[0]
    stop = time[-1]
    num = (stop-start)//D
    time_int = np.linspace(start, stop, num)
    fint = interpolate.interp1d(time, signal, 'cubic')
    signal_int = fint(time_int)
    return time_int, signal_int


data = np.loadtxt('ЭКГ.txt', delimiter='\t', dtype=np.float64)
data = data[174:574] #174-574; 574-1031; 1031-1313
filter = signal.firwin(10, [2, 70], fs=250, pass_zero=False)
filtered = signal.lfilter(filter, 1.0, data[:, 0])
trr = data[:, 1][data[:, 1] > 300]
trr = data[:, 1][data[:, 1] < 1000]
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
timeKNN = np.array(get_interpolation(timeNN,NN,100), dtype=int)
fig = plt.figure()
fig.set_figheight(20)
fig.set_figwidth(40)
plt.plot(timeNN/1000, NN, color='#08787f', marker='o', linewidth=2, markersize=5)
label_font = {'fontname': 'Times New Roman', 'size': '25'}
ticks_font = {'fontname': 'Times New Roman', 'size': '25'}
plt.xticks(**ticks_font)
plt.yticks(**ticks_font)
plt.title('Дворников Глеб ВСР')
plt.xlabel('Время, с', **label_font)
plt.ylabel('R-R, мс', **label_font)
plt.grid()
plt.savefig("photo.png")
#print(filtered)
plt.show()
M = np.mean(NN)
print('среднее -', M)
Hr = 1/M*60000
print('ЧСС -', Hr)
SDNN = np.std(NN)
print('SDNN(СКО) -', SDNN)
CV = SDNN/M*100
print('CV,% -', CV)
RMSDD = np.sqrt(sum(pow (np.diff(NN),2))/len (NN))
print('RMSDD -', RMSDD)
NN5O = len(np.diff(NN)[abs(np.diff(NN)>50)])
pNN5O = 100.0*len(np.diff(NN)[abs(np.diff(NN) >50)])/len(np.diff(NN))
print('NN5O -', NN5O)
print('pNN5O -', pNN5O)
