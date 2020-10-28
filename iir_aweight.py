from scipy.signal import chirp, freqz, firwin2, lfilter, bilinear_zpk, zpk2tf, zpk2sos, sosfreqz
import utils
import matplotlib.pyplot as plt
import numpy as np

RATE = 32000

z, p, k = bilinear_zpk([0.0, 0.0, 0.0, 0.0], [-129.4, -129.4, -676.7, -4636.0, -76655.0, -76655.0], 7.39705E9, RATE)
b, a = zpk2tf(z, p, k)
sos = zpk2sos(z, p, k)

coeff_array = []
print('numStages=%i' % sos.shape[0])
print()
for stage in sos:
    # print(stage)
    print('numerators/feedforward coefficients/b:')
    print(stage[:3])
    coeff_array = np.concatenate((coeff_array, stage[:3]))
    print('denominators/feedback coefficients/a:')
    print(stage[4:])
    coeff_array = np.concatenate((coeff_array, stage[4:]))
    print()

print(coeff_array)
# print('B taps:')
# for tap in b:
#     print('{:10.12f}'.format(tap))
#
# print('A taps:')
# for tap in a:
#     print('{:10.12f}'.format(tap))

exit()
# w, h = freqz(b, a, worN=32768)
w, h = sosfreqz(sos, worN=32768)
w_hz = w / max(w) * RATE / 2

utils.plot_resp(abs(h), w_hz, '-', '%i tap A weighting IIR filter' % len(a), meas_type='IIR/gain filter response', plot_in_db=True, xlim=[20, 20000], log_x=True)
plt.show()