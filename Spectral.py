import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy import signal
from scipy.fft import fft, fftfreq



def get_interpolation(time, signal, D):
    start = time[0]
    stop = time[-1]
    num = (stop-start)//D
    time_int = np.linspace(start, stop, num)
    fint = interpolate.interp1d(time, signal, 'cubic')
    signal_int = fint(time_int)
    return time_int, signal_int
def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n

def get_spectral_fourier(signal, D):
    l = nextpow2(signal.size)
    furie = np.fft.fft((signal - np.mean(signal)), n=l) / len(signal)
    Fs = 1 / D
    freq = (Fs / 2) * np.linspace(0, 1, int(1 + l / 2))
    furie = furie[0:len(freq)]
    return freq, furie



data = np.loadtxt('Вылегжанин Григорий.txt', delimiter='\t', dtype=np.float64)
data = data[:400] #0-400; 450-900; 950-1325

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
timeKNN = np.array(get_interpolation(timeNN,NN,1000)[0], dtype=int)

KNN = np.array(get_interpolation(timeNN,NN,1000)[1], dtype=float)

freq, furie = get_spectral_fourier(KNN, 1)
plt.plot(freq, abs(furie), color='#08787f', marker='o', linewidth=2, markersize=5)
label_font = {'fontname': 'Times New Roman', 'size': '15'}
ticks_font = {'fontname': 'Times New Roman', 'size': '15'}
plt.xticks(**ticks_font)
plt.yticks(**ticks_font)
plt.xlabel('Частота, Гц', **label_font)
plt.show()

f = (0.4, 0.15, 0.04, 0.003)
f1, f2, f3, f4 = f[0], f[1], f[2], f[3]
HF = (np.power(abs(furie[(freq > f2) * (freq < f1)]), 2))
HF_fr = sum(HF)
LF = (np.power(abs(furie[(freq > f3) * (freq < f2)]), 2))
LF_fr = sum(LF)
VLF = (np.power(abs(furie[(freq > f4) * (freq < f3)]), 2))
VLF_fr = sum(VLF)
TP_fr = HF_fr + LF_fr + VLF_fr
print('HF=', HF_fr)
print('LF=', LF_fr)
print('VLF=', VLF_fr)
print('TP_fr=', TP_fr)
LF_HF = LF_fr / HF_fr
print('LF/HF=', LF_HF)
IC = LF_fr + HF_fr / VLF_fr
print('IC=', IC)
IAS = LF_fr / VLF_fr
print('IAS=', IAS)