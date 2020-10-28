import wave
import numpy as np
from scipy.signal import lfilter
import struct
import matplotlib.pyplot as plt
import librosa as lb
from scipy.signal import chirp, freqz, firwin2, lfilter, bilinear_zpk, zpk2tf, zpk2sos, sosfreqz, sosfilt

RATE = 32000
BIT_RATE = 16
CHANNELS = 1
BLOCK_SIZE = (BIT_RATE // 8) * CHANNELS
MAX_INT_VAL = 2 ** BIT_RATE

in_data, sr = lb.load('in_chirp.wav', sr=RATE)

with open('filt_taps_fir.txt', 'r') as tap_file:
    taps = [float(i) for i in tap_file.readlines()]


out_data = lfilter(taps, [1.0], in_data / 4.3)

z, p, k = bilinear_zpk([0.0, 0.0, 0.0, 0.0], [-129.4, -129.4, -676.7, -4636.0, -76655.0, -76655.0], 7.39705E9, RATE)
b, a = zpk2tf(z, p, k)
sos = zpk2sos(z, p, k)
out_data = sosfilt(sos, in_data / 1.5)

wf = wave.open('filtered_chirp.wav', 'wb')
wf.setnchannels(CHANNELS)
wf.setframerate(RATE)
wf.setsampwidth(BLOCK_SIZE)

for i in range(len(out_data)):
    value = int(MAX_INT_VAL * out_data[i])
    data = struct.pack('<h', value)
    wf.writeframesraw(data)

wf.writeframes(b'')
wf.close()

nfft = len(out_data)
X = np.fft.rfft(out_data, n=nfft)
mag_spec_lin = [np.sqrt(i.real ** 2 + i.imag ** 2) / len(X) for i in X]
freq_arr = np.linspace(0, RATE / 2, num=len(mag_spec_lin))

plt.semilogx(freq_arr, 20 * np.log10(mag_spec_lin))
plt.show()

