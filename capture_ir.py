__author__ = 'Charlie'

import pyaudio
import time
import numpy as np
from scipy.signal import chirp, freqz, firwin2, lfilter, bilinear_zpk, zpk2tf
import utils
import matplotlib.pyplot as plt
import matplotlib
from platform import system
import seaborn as sns

sns.set(style="whitegrid")
matplotlib.rcParams['figure.figsize'] = (20.0, 8.0)
matplotlib.rcParams['axes.formatter.useoffset'] = False

CHANNELS = 1
RATE = 32000

# Locked at 16bits for now
BIT_RATE = 16
FRAME_BUF_LEN = 1024
BLOCK_SIZE = (BIT_RATE // 8) * CHANNELS

use_debug_files = True

p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')

if not use_debug_files:

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

    INPUT_DEVICE_ID = 3
    OUTPUT_DEVICE_ID = 2

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


def create_chirp(st_f, en_f, chirp_secs, method='logarithmic', boost_lo_freqs=False):
    """Create a chirp between frequencies st_f and en_f of duration chirp_secs.

        Parameters
        ----------
        method : str, (default is 'logarithmic')
            'logarithmic' or 'linear' style chirp
        boost_lo_freqs: bool, (default is False)
            Boost low frequencies below 500Hz if your reproduction system is a bit weedy

        Returns
        -------
        array
            Array of chirp samples at the projects sample rate

        """

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
        in_chirp_wav = utils.setup_wavfile('in_chirp.wav', CHANNELS, RATE, BLOCK_SIZE)
        in_chirp_wav.writeframes(chirp_y)
        in_chirp_wav.close()
    else:
        chirp_y = (
                chirp(t, st_f, chirp_secs, en_f, method=method, phi=90) * ((2 ** BIT_RATE) // 2)).astype(
            np.int16)
    return chirp_y


def measurement_cycle(meas_type, cycle, recreate_using_new_capture):
    global wav_file, samp_idx

    if not use_debug_files:
        wav_fname = '%s_sweep_%i.wav' % (meas_type, cycle)

        if recreate_using_new_capture:
            stream = setup_stream()

            wav_file = utils.setup_wavfile(wav_fname, CHANNELS, RATE, BLOCK_SIZE)
            create_chirp(st_freq, en_freq, sweep_secs, method='logarithmic')

            samp_idx = 0
            stream.start_stream()

            while stream.is_active():
                time.sleep(0.1)

            stream.close()
            wav_file.close()
    else:
        if meas_type == 'ref':
            wav_fname = 'audio/beh_simul.wav'
        else:
            wav_fname = 'audio/43434_simul.wav'

    mag_spec_y, freq_array, mag_spec_full, freqs_full = utils.ft_wavfile(wav_fname, fs=RATE)

    mag_spec_mat.append(mag_spec_y)

    # line_label = 'Cycle %i' % cycle
    # plot_resp(mag_spec_y, freq_array, '-', line_label, meas_type=meas_type, plot_in_db=True, xlim=[st_freq, en_freq])

    avg_mag_spec = utils.avg_resp(mag_spec_mat)

    utils.save_resp('%s_resp.h5' % meas_type, [avg_mag_spec, freq_array])

    print('\tCompleted capture of %s' % wav_fname)

    return mag_spec_mat, freq_array


# TODO: CHANGE TO 3
cycle_cnt = 3
octave_smooth = 24

# Should be at least 10s
sweep_secs = 10

st_freq = 20
en_freq = RATE // 2

mag_spec_mat = []

boost_lo_freqs = True

# recreate_ref_h5_from_wavs = True
# recreate_dut_h5_from_wavs = True
recreate_ref = False
recreate_dut = False

ref_resp_fname = 'ref_resp.h5'

# if not os.path.isfile(ref_resp_fname) or recreate_ref:
print('Starting ref capture process ...')
for x in range(1, cycle_cnt + 1):
    mag_spec_avg_ref, freqs = measurement_cycle('ref', x, recreate_ref)
p.terminate()

mag_spec_avg_ref = utils.avg_resp(mag_spec_avg_ref)

# plot_resp(mag_spec_avg_ref, freqs, '--', 'Median', meas_type='ref', plot_in_db=True, xlim=[st_freq, en_freq])
# plt.legend()
# plt.show()
# plt.close()
# else:
#     with h5py.File(ref_resp_fname, 'r') as f:
#         mag_spec_avg_ref = f['mag_spec_avg'][()][0]
#         freqs = f['mag_spec_avg'][()][1]


dut_resp_fname = 'dut_resp.h5'

# if not os.path.isfile(dut_resp_fname) or recreate_dut:
p = pyaudio.PyAudio()
if not use_debug_files:
    for i in range(0, numdevices):
        if p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') > 0:
            print("Input device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
    # INPUT_DEVICE_ID = int(input("Enter the device id to use for dut audio input: "))
    INPUT_DEVICE_ID = 3
    print()

mag_spec_mat = []

print('Starting dut capture process ...')
for x in range(1, cycle_cnt + 1):
    mag_spec_avg_dut, freqs = measurement_cycle('dut', x, recreate_dut)

p.terminate()

mag_spec_avg_dut = utils.avg_resp(mag_spec_avg_dut)
# else:
#     with h5py.File(dut_resp_fname, 'r') as f:
#         mag_spec_avg_dut = f['mag_spec_avg'][()][0]
#         freqs = f['mag_spec_avg'][()][1]

mag_spec_avg_ref_db = 20 * np.log10(mag_spec_avg_ref)
mag_spec_avg_dut_db = 20 * np.log10(mag_spec_avg_dut)

# mag_spec_avg_ref_db, mag_spec_avg_dut_db = align_resps(mag_spec_avg_ref_db, mag_spec_avg_dut_db, freqs)

utils.plot_resp(mag_spec_avg_ref_db, freqs, '-', 'Reference', meas_type='Ref resp', plot_in_db=False, xlim=[st_freq, en_freq], log_x=True)
utils.plot_resp(mag_spec_avg_dut_db, freqs, '-', 'DUT', meas_type='Dut resp', plot_in_db=False, xlim=[st_freq, en_freq], log_x=True)
plt.legend()
plt.show()

# diff_resp = sub_resps(mag_spec_avg_ref_db, mag_spec_avg_dut_db, freqs)
diff_resp = mag_spec_avg_ref_db - mag_spec_avg_dut_db

utils.plot_resp(diff_resp, freqs, '-', 'Diffs', meas_type='diff', plot_in_db=False, xlim=[st_freq, en_freq], log_x=True)
plt.legend()
plt.show()




filt_gains_linear = 10**(diff_resp/20)

st_freq = 1
en_freq = RATE // 2

plot_log = True

freq_filt_str = 'none'

filt_gains_clean, filt_gains_freq = utils.cleanup_desired_filter_gain(filt_gains_linear.copy(), freqs.copy(), freq_range=freq_filt_str, fs=RATE)

# plot_resp(filt_gains_linear, freqs, '-', 'Filter gain', meas_type='filter gains', plot_in_db=True, xlim=[st_freq, en_freq], log_x=plot_log)
# plot_resp(filt_gains_clean, filt_gains_freq, '-', 'Cleaned filter gain', meas_type='diff', plot_in_db=True, xlim=[st_freq, en_freq], log_x=plot_log)

numtaps_fir = 1025
taps = firwin2(numtaps_fir, filt_gains_freq, filt_gains_clean, fs=RATE)
for tap in taps:
    print('{:10.24f}'.format(tap))
w, h = freqz(taps, 1.0, worN=32768)
w_hz = w / max(w) * RATE / 2

with open('filt_taps_fir.txt', 'w') as out_file:
    for tap in taps:
        out_file.write('{:10.64f}'.format(tap))
        out_file.write('\n')


in_data = [0] * RATE

in_data[0] = len(in_data) / 2

out_data = np.convolve(taps, in_data, mode='valid')
out_data = lfilter(taps, [1.0], in_data)

nfft = len(out_data)
X = np.fft.rfft(out_data, n=nfft)
mag_spec_lin = [np.sqrt(i.real ** 2 + i.imag ** 2) / len(X) for i in X]
freq_arr = np.linspace(0, RATE / 2, num=len(mag_spec_lin))
mag_spec_avg, freqs_avg = utils.octsmooth(mag_spec_lin, freq_arr, octave_smooth)

plt.semilogx(freq_arr, 20*np.log10(mag_spec_lin))
plt.show()

utils.plot_resp(abs(h), w_hz, '-', '%i tap FIR filter' % numtaps_fir, meas_type='IIR/gain filter response', plot_in_db=True, xlim=[st_freq, en_freq], log_x=plot_log)

numtaps_iir = 55
b, a = utils.yulewalk(numtaps_iir, filt_gains_freq / np.max(filt_gains_freq), filt_gains_clean)
w, h = freqz(b, a, worN=32768)
w_hz = w / max(w) * RATE / 2

utils.plot_resp(abs(h), w_hz, '-', '%i tap IIR filter' % numtaps_iir, meas_type='IIR/gain filter response', plot_in_db=True, xlim=[st_freq, en_freq], log_x=plot_log)
plt.legend()
# plt.ylim([-2, 10])
plt.savefig('plot_%s_iir-%i_fir-%i.png' % (freq_filt_str, numtaps_iir, numtaps_fir))
plt.show()


z, p, k = bilinear_zpk([0.0, 0.0, 0.0, 0.0], [-129.4, -129.4, -676.7, -4636.0, -76655.0, -76655.0], 7.39705E9, RATE)
b, a = zpk2tf(z, p, k)

print('B taps:')
for tap in b:
    print('{:10.12f}'.format(tap))

print('A taps:')
for tap in a:
    print('{:10.12f}'.format(tap))

w, h = freqz(b, a, worN=32768)
w_hz = w / max(w) * RATE / 2

utils.plot_resp(abs(h), w_hz, '-', '%i tap A weighting IIR filter' % numtaps_iir, meas_type='IIR/gain filter response', plot_in_db=True, xlim=[st_freq, en_freq], log_x=plot_log)
plt.show()