import wave
import numpy as np
from scipy.signal import lfilter
import struct
import matplotlib.pyplot as plt

RATE = 48000
BIT_RATE = 16
CHANNELS = 1
BLOCK_SIZE = (BIT_RATE // 8) * CHANNELS
MAX_INT_VAL = 2 ** BIT_RATE

wf = wave.open('in_chirp.wav', 'r')

wf_bytes = wf.readframes(wf.getnframes())

wf.close()

in_data = np.frombuffer(wf_bytes, np.int16) / (MAX_INT_VAL // 2)

with open('filt_taps_fir.txt', 'r') as tap_file:
    taps = [float(i) for i in tap_file.readlines()]


out_data = lfilter(taps, [1.0], in_data / 4.0)

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

