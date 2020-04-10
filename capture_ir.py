__author__ = 'Charlie'

import pyaudio

import time
import wave
import numpy as np
import h5py
from scipy.signal import chirp, freqz, firwin2
import os
import utils
from acoustics.octave import Octave
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import ScalarFormatter
from platform import system
import seaborn as sns

sns.set(style="whitegrid")
matplotlib.rcParams['figure.figsize'] = (20.0, 8.0)
matplotlib.rcParams['axes.formatter.useoffset'] = False

CHANNELS = 1
RATE = 48000

# Locked at 16bits for now
BIT_RATE = 16
FRAME_BUF_LEN = 1024
BLOCK_SIZE = (BIT_RATE // 8) * CHANNELS

p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')

for i in range(0, numdevices):
    if p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') > 0:
        print("Input device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
INPUT_DEVICE_ID = int(input("Enter the device id to use for audio input: "))
print()

for i in range(0, numdevices):
    if p.get_device_info_by_host_api_device_index(0, i).get('maxOutputChannels') > 0:
        print("Output device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
OUTPUT_DEVICE_ID = int(input("Enter the device id to use for audio output: "))

# INPUT_DEVICE_ID = 2
# OUTPUT_DEVICE_ID = 2

wav_file = None
samp_idx = 0
chirp_y = []


def plt_maximize():
    # See discussion: https://stackoverflow.com/questions/12439588/how-to-maximize-a-plt-show-window-using-python
    backend = plt.get_backend()
    cfm = plt.get_current_fig_manager()
    if backend == "wxAgg":
        cfm.frame.Maximize(True)
    elif backend == "TkAgg":
        if system() == "win32":
            cfm.window.state('zoomed')  # This is windows only
        else:
            cfm.resize(*cfm.window.maxsize())
    elif backend == 'QT4Agg':
        cfm.window.showMaximized()
    elif callable(getattr(cfm, "full_screen_toggle", None)):
        if not getattr(cfm, "flag_is_max", None):
            cfm.full_screen_toggle()
            cfm.flag_is_max = True
    else:
        raise RuntimeError("plt_maximize() is not implemented for current backend:", backend)


def stream_callback(in_data, frame_count, time_info, status):
    global samp_idx
    global wav_file
    wav_file.writeframes(in_data)
    out_data = chirp_y[samp_idx:samp_idx + frame_count]
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


def create_chirp(st_f, en_f, chirp_secs, method):
    global chirp_y
    t = np.arange(0, chirp_secs, 1 / RATE)
    t_f = np.geomspace(0.001, RATE / 2, num=len(t))

    if boost_lo_freqs:
        chirp_y = (
                (chirp(t, st_f, chirp_secs, en_f, method=method, phi=90) / 2.0) * ((2 ** BIT_RATE) // 2)).astype(
            np.int16)
        max_freq = 500
        max_freq_idx = np.abs(t_f - max_freq).argmin()
        lf_window = np.hanning(max_freq_idx * 2)
        lf_window = lf_window[max_freq_idx - 1:-1] + 1.0
        chirp_y[0:max_freq_idx] = chirp_y[0:max_freq_idx] * lf_window
        in_chirp_wav = setup_wavfile('in_chirp.wav')
        in_chirp_wav.writeframes(chirp_y)
        in_chirp_wav.close()
    else:
        chirp_y = (
                chirp(t, st_f, chirp_secs, en_f, method=method, phi=90) * ((2 ** BIT_RATE) // 2)).astype(
            np.int16)
    return chirp_y


def ft_wavfile(wave_fname):
    wf = wave.open(wave_fname, 'r')
    wav_bytes = wf.readframes(wf.getnframes())
    y = np.frombuffer(wav_bytes, np.int16) / ((2 ** BIT_RATE) // 2)

    # plt.plot(y)
    # plt.show()
    # plt.close()
    nfft = len(y)
    X = np.fft.rfft(y, n=nfft)
    mag_spec_lin = [np.sqrt(i.real ** 2 + i.imag ** 2) / len(X) for i in X]
    freq_arr = np.linspace(0, RATE / 2, num=len(mag_spec_lin))
    mag_spec_avg, freqs_avg = octsmooth(mag_spec_lin, freq_arr, octave_smooth)
    return mag_spec_avg, freqs_avg, mag_spec_lin, freq_arr


def plot_resp(mag_spec_oct, freqs_oct, linestyle, legend_label, plot_in_db=True, meas_type='', xlim=[0, 0], log_x=True):
    if log_x:
        if plot_in_db:
            plt.semilogx(freqs_oct, 20 * np.log10(mag_spec_oct), linestyle, label=legend_label)
        else:
            plt.semilogx(freqs_oct, mag_spec_oct, linestyle, label=legend_label)
    else:
        if plot_in_db:
            plt.plot(freqs_oct, 20 * np.log10(mag_spec_oct), linestyle, label=legend_label)
        else:
            plt.plot(freqs_oct, mag_spec_oct, linestyle, label=legend_label)
    # cax = plt.gca().xaxis
    # cax.set_major_formatter(ScalarFormatter())
    # cay = plt.gca().yaxis
    # cay.set_major_formatter(ScalarFormatter())

    plt.xlim(xlim)
    plt.title('Frequency response of %s' % meas_type)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.tight_layout()


def octsmooth(amps, freq_vals, noct):
    o = Octave(fmin=st_freq, fmax=en_freq, fraction=noct)
    octbins = np.zeros(len(o.center))
    for i in range(0, len(o.center)):
        st = (np.abs(freq_vals - o.lower[i])).argmin()
        en = (np.abs(freq_vals - o.upper[i])).argmin()
        if en - st > 0:
            octbinvec = amps[st:en]
        else:
            octbinvec = amps[st:en + 1]
        octbins[i] = np.max(octbinvec)
    return octbins, o.center


def avg_resp(mag_mat):
    return np.median(np.asarray(mag_mat), axis=0)


def save_resp(fname, data):
    hf = h5py.File(fname, 'w')
    hf.create_dataset('mag_spec_avg', data=data)
    hf.close()


def measurement_cycle(meas_type, cycle):
    global wav_file, samp_idx

    wav_fname = '%s_sweep_%i.wav' % (meas_type, cycle)

    stream = setup_stream()

    wav_file = setup_wavfile(wav_fname)
    create_chirp(st_freq, en_freq, sweep_secs, 'logarithmic')

    samp_idx = 0
    stream.start_stream()

    while stream.is_active():
        time.sleep(0.1)

    stream.close()
    wav_file.close()

    # # TODO: REMOVE - JUST FOR DEBUGGING
    # wav_fname = 'audio/43434_simul.wav'
    mag_spec_y, freq_array, mag_spec_full, freqs_full = ft_wavfile(wav_fname)

    mag_spec_mat.append(mag_spec_y)

    line_label = 'Cycle %i' % cycle
    plot_resp(mag_spec_y, freq_array, '-', line_label, meas_type=meas_type, plot_in_db=True, xlim=[st_freq, en_freq])

    avg_mag_spec = avg_resp(mag_spec_mat)

    save_resp('%s_resp.h5' % meas_type, [avg_mag_spec, freq_array])

    print('\tCompleted capture of %s' % wav_fname)

    return avg_resp(mag_spec_mat), freq_array


def align_resps(ref_mags, dut_mags, f_arr):
    align_freq = 1000
    freq_idx = np.abs(f_arr - align_freq).argmin()
    return ref_mags - ref_mags[freq_idx], dut_mags - dut_mags[freq_idx]


def sub_resps(ref_mags, dut_mags, f_arr):
    diff_mags = ref_mags - dut_mags
    align_freq = 1000
    freq_idx = np.abs(f_arr - align_freq).argmin()
    return diff_mags - diff_mags[freq_idx]


def cleanup_desired_filter_gain(orig_filt_gains, orig_filt_freqs):

    orig_filt_freqs[0] = 0

    if RATE / 2 > orig_filt_freqs[-1]:
        np.append(orig_filt_freqs, RATE / 2)
        np.append(orig_filt_gains, orig_filt_gains[-1])

    if RATE / 2 < orig_filt_freqs[-1]:
        orig_filt_freqs[-1] = RATE / 2

    lo_align_freq = 20
    lo_freq_idx = np.abs(orig_filt_freqs - lo_align_freq).argmin()

    orig_filt_gains[0:lo_freq_idx] = orig_filt_gains[lo_freq_idx]

    return orig_filt_gains, orig_filt_freqs


# TODO: CHANGE TO 3
cycle_cnt = 3
octave_smooth = 24

# Should be at least 10s
sweep_secs = 10

st_freq = 1
en_freq = RATE // 2

mag_spec_mat = []

boost_lo_freqs = False

recreate_ref = True

ref_resp_fname = 'ref_resp.h5'

if not os.path.isfile(ref_resp_fname) or recreate_ref:
    print('Starting ref capture process ...')
    for x in range(1, cycle_cnt + 1):
        mag_spec_avg_ref, freqs = measurement_cycle('ref', x)
    p.terminate()

    plot_resp(mag_spec_avg_ref, freqs, '--', 'Median', meas_type='ref', plot_in_db=True, xlim=[st_freq, en_freq])
    plt.legend()
    plt.show()
    plt.close()
else:
    with h5py.File(ref_resp_fname, 'r') as f:
        mag_spec_avg_ref = f['mag_spec_avg'][()][0]
        freqs = f['mag_spec_avg'][()][1]

recreate_dut = True

dut_resp_fname = 'dut_resp.h5'

if not os.path.isfile(dut_resp_fname) or recreate_dut:
    p = pyaudio.PyAudio()
    for i in range(0, numdevices):
        if p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') > 0:
            print("Input device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
    INPUT_DEVICE_ID = int(input("Enter the device id to use for dut audio input: "))
    # INPUT_DEVICE_ID = 3
    print()

    mag_spec_mat = []

    print('Starting dut capture process ...')
    for x in range(1, cycle_cnt + 1):
        mag_spec_avg_dut, freqs = measurement_cycle('dut', x)

    p.terminate()
else:
    with h5py.File(dut_resp_fname, 'r') as f:
        mag_spec_avg_dut = f['mag_spec_avg'][()][0]
        freqs = f['mag_spec_avg'][()][1]

mag_spec_avg_ref_db = 20 * np.log10(mag_spec_avg_ref)
mag_spec_avg_dut_db = 20 * np.log10(mag_spec_avg_dut)

mag_spec_avg_ref_db, mag_spec_avg_dut_db = align_resps(mag_spec_avg_ref_db, mag_spec_avg_dut_db, freqs)

diff_resp = sub_resps(mag_spec_avg_ref_db, mag_spec_avg_dut_db, freqs)

filt_gains = 10**(diff_resp/20)

# st_freq = -10
# en_freq = 5000
st_freq = 1
en_freq = RATE // 2

plot_log = True

filt_gains_lin, filt_gains_freq = cleanup_desired_filter_gain(diff_resp, freqs)

plot_resp(filt_gains, freqs, '-', 'Ideal filter gain', meas_type='filter gains', plot_in_db=True, xlim=[st_freq, en_freq], log_x=plot_log)

numtaps = 1025
taps = firwin2(numtaps, filt_gains_freq, filt_gains_lin, fs=RATE)
w, h = freqz(taps, 1.0, worN=32768)
w_hz = w / max(w) * RATE / 2

plot_resp(abs(h), w_hz, '-', '%i tap FIR filter' % numtaps, meas_type='IIR/gain filter response', plot_in_db=False, xlim=[st_freq, en_freq], log_x=plot_log)

numtaps = 75

b, a = utils.yulewalk(numtaps, filt_gains_freq / np.max(filt_gains_freq), filt_gains_lin)
w, h = freqz(b, a, worN=32768)
w_hz = w / max(w) * RATE / 2

plot_resp(abs(h), w_hz, '-', '%i tap IIR filter' % numtaps, meas_type='IIR/gain filter response', plot_in_db=False, xlim=[st_freq, en_freq], log_x=plot_log)
plt.legend()
plt.show()


