__author__ = 'Charlie'

import pyaudio

import time
import wave
import numpy as np
from scipy.signal import chirp
from acoustics.octave import Octave
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

CHANNELS = 1
RATE = 48000

# Locked at 16bits for now
BIT_RATE = 16
FRAME_BUF_LEN = 1024
BLOCK_SIZE = (BIT_RATE // 8) * CHANNELS

p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')

# for i in range(0, numdevices):
#     if p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') > 0:
#         print("Input device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
# INPUT_DEVICE_ID = int(input("Enter the device id to use for audio input: "))
# print()
#
# for i in range(0, numdevices):
#     if p.get_device_info_by_host_api_device_index(0, i).get('maxOutputChannels') > 0:
#         print("Output device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
# OUTPUT_DEVICE_ID = int(input("Enter the device id to use for audio output: "))


# INPUT_DEVICE_ID = 0
# OUTPUT_DEVICE_ID = 1

INPUT_DEVICE_ID = 2
OUTPUT_DEVICE_ID = 2

st_freq = 1
en_freq = RATE // 2

# Should be at least 10s
sweep_secs = 10

t = np.arange(0, sweep_secs, 1 / RATE)

log_chirp_y = (chirp(t, st_freq, sweep_secs, en_freq, method='logarithmic', phi=90) * ((2 ** BIT_RATE) // 2)).astype(
    np.int16)


def stream_callback(in_data, frame_count, time_info, status):
    global samp_idx
    global wav_file
    wav_file.writeframes(in_data)
    out_data = log_chirp_y[samp_idx:samp_idx + frame_count]
    samp_idx += frame_count
    return out_data, pyaudio.paContinue


def setup_stream():
    return p.open(format=p.get_format_from_width(BLOCK_SIZE),
                  channels=CHANNELS,
                  rate=RATE,
                  input=True,
                  output=True,
                  stream_callback=stream_callback,
                  input_device_index=INPUT_DEVICE_ID,
                  output_device_index=OUTPUT_DEVICE_ID,
                  frames_per_buffer=FRAME_BUF_LEN)


def setup_wavfile(fname):
    wf = wave.open(fname, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setframerate(RATE)
    wf.setsampwidth(BLOCK_SIZE)
    return wf


def ft_wavfile(wave_fname):
    wf = wave.open(wave_fname, 'r')
    wav_bytes = wf.readframes(wf.getnframes())
    y = np.frombuffer(wav_bytes, np.int16) / ((2 ** BIT_RATE) // 2)
    nfft = len(y)
    X = np.fft.rfft(y, n=nfft)
    # mag_spec = 20 * np.log10([np.sqrt(i.real ** 2 + i.imag ** 2) / len(X) for i in X])
    mag_spec_lin = [np.sqrt(i.real ** 2 + i.imag ** 2) / len(X) for i in X]
    freqs = np.linspace(0, RATE / 2, num=len(mag_spec_lin))
    mag_spec_avg, freqs_avg = octsmooth(mag_spec_lin, freqs, octave_smooth)
    plt.semilogx(freqs_avg, 20 * np.log10(mag_spec_avg))
    plt.semilogx(freqs, 20 * np.log10(mag_spec_lin))
    plt.xlim([st_freq, en_freq])


def octsmooth(amps, freq_array, noct):
    o = Octave(fmin=st_freq, fmax=en_freq, fraction=noct)
    octbins = np.zeros(len(o.center))
    for i in range(0, len(o.center)):
        st = (np.abs(freq_array - o.lower[i])).argmin()
        en = (np.abs(freq_array - o.upper[i])).argmin()
        if en - st > 0:
            octbinvec = amps[st:en]
        else:
            octbinvec = amps[st:en + 1]
        octbins[i] = np.max(octbinvec)
    return octbins, o.center


cycle_cnt = 3
octave_smooth = 12

for x in range(1, cycle_cnt + 1):
    wav_fname = 'ir_%i.wav' % x
    stream = setup_stream()
    wav_file = setup_wavfile(wav_fname)
    samp_idx = 0
    stream.start_stream()

    while stream.is_active():
        time.sleep(0.1)

    stream.close()
    wav_file.close()
    ft_wavfile(wav_fname)

plt.show()
p.terminate()
