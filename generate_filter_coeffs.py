import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

filename = 'audio/beh_simul.wav'

file_sr = librosa.get_samplerate(filename)
y, sr = librosa.load(filename, sr=file_sr)

nfft = len(y)

X = np.fft.rfft(y, n=nfft)

mag_spec_ref = 20 * np.log10([np.sqrt(i.real ** 2 + i.imag ** 2) / len(X) for i in X])

freqs = np.linspace(0, sr / 2, num=len(mag_spec_ref))

plt.semilogx(freqs, mag_spec_ref)
plt.xlim([1, 24000])
plt.show()
