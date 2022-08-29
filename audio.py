import numpy as np
import matplotlib.pyplot as plt
# from scipy.io.wavfile import read, write
import librosa
import statsmodels.api as sm
# from IPython.display import Audio
from scipy.signal import find_peaks
from numpy.fft import fft, ifft

data, sampling_frequency = librosa.load("audios/roy.wav")

T = 1 / sampling_frequency
N = len(data)
t = N / sampling_frequency

Y_k = fft(data)[0:int(N/2)]/N
Y_k[1:] = 2*Y_k[1:]
Pxx = np.abs(Y_k)

f = sampling_frequency * np.arange((N/2)) / N

fig, ax = plt.subplots()
plt.plot(f[0:10000], Pxx[0:10000], linewidth=2)
plt.ylabel("Amplitude")
plt.xlabel("Frequency")
# plt.xticks(np.arange(500, 10))
plt.grid()

auto = sm.tsa.acf(data, nlags=2000)
peaks = find_peaks(auto)[0]
lag = peaks[0]

pitch = sampling_frequency / lag

print(pitch)
plt.show()

'''
fs, data = read("F:\\audio\\400Hz.wav")

# data = data[:, 0]
print(f"Sampling frequency = {fs}")

ft = fft(data)

plt.figure()
plt.plot(np.abs(ft))
plt.xlabel("Sample Index")
plt.ylabel("fourier")
plt.title("Waveform of Test Audio")

plt.show()'''
