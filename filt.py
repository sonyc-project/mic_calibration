from scipy.signal import lfilter

with open('filt_taps_fir.txt', 'r') as tap_file:
    taps = [float(i) for i in tap_file.readlines()]

in_data = []

out_data = lfilter(taps, [1.0], in_data)
