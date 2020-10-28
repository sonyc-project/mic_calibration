import numpy as np
from scipy import signal, linalg
import h5py
import wave
from acoustics.octave import Octave
import matplotlib.pyplot as plt
import librosa as lb


def setup_wavfile(fname, ch_num=1, fs=44100, bs=2):
    """Setup wave file for writing.

    Args
    ----------
    fname : str
        The filename of the output wave file

    Parameters
    ----------
    ch_num : int
        Number of channels (default is 1)

    fs : int
        Sample rate in Hz (default is 44100)

    bs : int
        Block size (default is 2)

    Returns
    -------
    Wave_write
        A file like object to write bytes to a wave file

    """

    wf = wave.open(fname, 'wb')
    wf.setnchannels(ch_num)
    wf.setframerate(fs)
    wf.setsampwidth(bs)
    return wf


def ft_wavfile(wave_fname, fs=44100, br=16, octave_smooth=24):
    """DFT a wave file.

    Args
    ----------
    wave_fname : str
        Path to wave file

    Parameters
    ----------
    ch_num : int
        Number of channels (default is 1)

    fs : int
        Sample rate in Hz (default is 44100)

    br : int
        Bit rate (default is 16)

    octave_smooth : int
        Octave smoothing amount (default is 24)

    Returns
    -------
    array
        Octave smoothed magnitude spectrum array
    array
        Frequency values for each magnitude spectrum value
    array
        Non octave smoothed agnitude spectrum values
    array
        Non octave smoothed frequency values for each magnitude spectrum value

    """
    y, sr = lb.load(wave_fname, sr=fs)
    # wf = wave.open(wave_fname, 'r')
    # wav_bytes = wf.readframes(wf.getnframes())
    # y = np.frombuffer(wav_bytes, np.int16) / ((2 ** br) // 2)

    nfft = len(y)
    X = np.fft.rfft(y, n=nfft)
    mag_spec_lin = [np.sqrt(i.real ** 2 + i.imag ** 2) / len(X) for i in X]
    freq_arr = np.linspace(0, fs / 2, num=len(mag_spec_lin))
    mag_spec_avg, freqs_avg = octsmooth(mag_spec_lin, freq_arr, noct=octave_smooth, st_freq=20, en_freq=fs/2)
    return mag_spec_avg, freqs_avg, mag_spec_lin, freq_arr


def plot_resp(mag_spec_oct, freqs_oct, linestyle='-', legend_label='', plot_in_db=True, meas_type='', xlim=[20, 20000], log_x=True):
    """Plot frequency response.

        Args
        ----------
        mag_spec_oct : array
            Magnitude spectrum values

        freqs_oct : array
            Magnitude spectrum frequency values

        Parameters
        ----------
        linestyle : str, optional
            Plot linestyle (default is '-')

        legend_label : str, optional
            Line legend label (default is '')

        plot_in_db : bool, optional
            Plot the response in decibels (default is True)

        meas_type : str, optional
            String to add to title to identify type of plot (usually 'diff' or 'ref')

        xlim : array
            X axis limits in the form [num, num] (default is [0, 0])

        log_x : bool, optional
            Logarithmic X axis (default is True)

        """
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

    plt.xlim(xlim)
    plt.title('Frequency response of %s' % meas_type)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.tight_layout()


def octsmooth(amps, freq_vals, noct=24, st_freq=20, en_freq=20000):
    """Smooth magnitude spectrum values using octave bands for easier subtractions and viz.

    Args
    ----------
    amps : array
        Magnitude spectrum values for smoothing

    freq_vals : array
        Magnitude spectrum frequency values

    Parameters
    ----------
    noct : int, optional
        Number of octaves to smooth over (default is 24)

    st_freq : int, optional
        Start frequency for octave bands (default is 20)

    en_freq : int, optional
        End frequency for octave bands (default is 20000)

    Returns
    -------
    array
        Octave smoothed magnitude spectrum array
    array
        Octave smoothed magnitude spectrum array frequency bins
            """
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
    """Average response using a median across multiple responses in matrix.
    Median is used to reduce the impact of an especially weird and noisy measurement
    that could swing your average out.

    Args
    ----------
    mag_mat: mat
        N Magnitude spectrum values in matrix

    Returns
    -------
    array
        Median averaged array
    """
    return np.median(np.asarray(mag_mat), axis=0)


def save_resp(fname, data):
    """Save averaged response to H5 file.

    Args
    ----------
    fname: str
        Filename of output H5 file

    data: array
        Averaged magnitude spectrum data

    """
    hf = h5py.File(fname, 'w')
    hf.create_dataset('mag_spec_avg', data=data)
    hf.close()


def align_resps(ref_mags, dut_mags, f_arr, align_freq=1000):
    """Shift reference and DUT responses to meet at a given frequency.

    Args
    ----------
    ref_mags: array
        Magnitude spectrum values of reference device

    dut_mags: array
        Magnitude spectrum values of device under test (DUT)

    f_arr: array
        Frequency array for magnitude values

    Parameters
    ----------
    align_freq : int, optional
        Alignment frequency (default is 1000)

    Returns
    -------
    array
        Aligned responses
    """

    freq_idx = np.abs(f_arr - align_freq).argmin()
    return ref_mags - ref_mags[freq_idx], dut_mags - dut_mags[freq_idx]


def sub_resps(ref_mags, dut_mags, f_arr, align_freq=1000):
    """Subtract responses, performing alignment to given frequency.

    Args
    ----------
    ref_mags: array
        Magnitude spectrum values of reference device

    dut_mags: array
        Magnitude spectrum values of device under test (DUT)

    f_arr: array
        Frequency array for magnitude values

    Parameters
    ----------
    align_freq : int, optional
        Alignment frequency (default is 1000)

    Returns
    -------
    array
        Aligned responses
    """
    diff_mags = ref_mags - dut_mags
    freq_idx = np.abs(f_arr - align_freq).argmin()
    return diff_mags - diff_mags[freq_idx]


def cleanup_desired_filter_gain(orig_filt_gains=[], orig_filt_freqs=[], lo_align_freq=40, freq_range='low', fs=44100):
    """Cleans up a filter gain array at the low frequency range by levelling out low
    frequencies below a given frequency to help in filter design by removing
    extreme gain shifts.

    Parameters
    ----------
    orig_filt_gains : array
        The original filter gains to be cleaned (default is [])

    orig_filt_freqs : array
        The original filter gain's frequency bins (default is [])

    lo_align_freq : int, optional
        Low frequency to end leveling at (default is 40)

    freq_range: str, optional
        If 'high' is passed this will level out all gains below 1000Hz (default is 'low')

    fs: int, optional
        Sample rate in Hz (default is 44100)
    """

    orig_filt_freqs[0] = 0

    if fs / 2 > orig_filt_freqs[-1]:
        orig_filt_freqs = np.append(orig_filt_freqs, fs / 2)
        orig_filt_gains = np.append(orig_filt_gains, orig_filt_gains[-1])

    if fs / 2 < orig_filt_freqs[-1]:
        orig_filt_freqs[-1] = fs / 2

    mid_align_freq = 1000
    mid_freq_idx = np.abs(orig_filt_freqs - mid_align_freq).argmin()

    if freq_range == 'high':
        orig_filt_gains[0:mid_freq_idx] = orig_filt_gains[mid_freq_idx]
    elif freq_range == 'low':
        orig_filt_gains[mid_freq_idx:-1] = orig_filt_gains[mid_freq_idx]
        orig_filt_gains[-1] = orig_filt_gains[mid_freq_idx]

    lo_freq_idx = np.abs(orig_filt_freqs - lo_align_freq).argmin()
    orig_filt_gains[0:lo_freq_idx] = orig_filt_gains[lo_freq_idx]

    return orig_filt_gains, orig_filt_freqs


# Source: https://github.com/mrazavian/ITURPropagPY/blob/18eac5b66b95b868d23a93977872ae9fe6bbaac4/iturpropag/models/iturp1853/scintillation_attenuation_synthesis.py


def yulewalk(na, ff, aa):
    """
    YULEWALK Recursive filter design using a least-squares method.
    `[b,a] = yulewalk(na,ff,aa)` finds the `na-th` order recursive filter
    coefficients `b` and `a` such that the filter::
                            -1              -M
                b[0] + b[1]z  + ... + b[M] z
        Y(z) = -------------------------------- X(z)
                            -1              -N
                a[0] + a[1]z  + ... + a[N] z
    matches the magnitude frequency response given by vectors `ff` and `aa`.
    Vectors `ff` and `aa` specify the frequency and magnitude breakpoints for
    the filter such that `plot(ff,aa)` would show a plot of the desired
    frequency response.
    
    Parameters
    -----------
    - na : integer scalar.
            order of the recursive filter
    - ff : 1-D array
            frequencies sampling which must be between 0.0 and 1.0,
            with 1.0 corresponding to half the sample rate. They must be in
            increasing order and start with 0.0 and end with 1.0.
    
    Returns
    -----------
    - b : 1-D array
            numerator coefficients of the recursive filter
    - a : 1-D array
            denumerator coefficients of the recursive filter
    
    References
    ------------
    [1] Friedlander, B., and Boaz Porat. "The Modified Yule-Walker Method of 
    ARMA Spectral Estimation." IEEE® Transactions on Aerospace Electronic 
    Systems. Vol. AES-20, Number 2, 1984, pp. 158–173.
    [2] Matlab R2016a `yulewalk` function
    """
    npt = 512
    lap = np.fix(npt / 25)

    npt = npt + 1
    Ht = np.zeros(npt)

    nint = np.size(ff) - 1
    df = np.diff(ff)

    nb = 1
    Ht[0] = aa[0]
    for ii in np.arange(nint):

        if df[ii] == 0:
            nb = int(nb - lap / 2)
            ne = int(nb + lap)
        else:
            ne = int(np.fix(ff[ii + 1] * npt))

        jj = np.arange(nb, ne + 1)
        if ne == nb:
            inc = 0
        else:
            inc = (jj - nb) / (ne - nb);

        Ht[nb - 1: ne] = inc * aa[ii + 1] + (1 - inc) * aa[ii]
        nb = int(ne + 1)

    Ht = np.append(Ht, Ht[-2:0:-1])
    n = np.size(Ht)
    n2 = int(np.fix((n + 1) / 2))
    nb = na
    nr = 4 * na
    nt = np.arange(nr)

    R = np.real(np.fft.ifft(Ht * Ht))
    R = R[:nr] * (0.54 + 0.46 * np.cos(np.pi * nt / (nr - 1)))

    Rwindow = np.append(0.5, np.ones(n2 - 1))
    Rwindow = np.append(Rwindow, np.zeros(n - n2))

    A = polystab(denf(R, na))

    R = R[:nr]
    R[0] = R[0] / 2
    Qh = numf(R, A, na)

    _, Ss = 2 * np.real(signal.freqz(Qh, A, n, whole=True))
    var1 = np.log(Ss.astype('complex'))
    var2 = np.fft.ifft(var1)
    hh = np.fft.ifft(np.exp(np.fft.fft(Rwindow * var2)))
    B = np.real(numf(hh[:nr], A, nb))

    return B, A


def polystab(a):
    """
    Polynomial stabilization.
    polystab(a), where a is a vector of polynomial coefficients,
    stabilizes the polynomial with respect to the unit circle;
    roots whose magnitudes are greater than one are reflected
    inside the unit circle.
    Parameters
    ----------
    - a : 1-D numpy.array.
            vector of polynomial coefficients
    
    Returns
    -------
    - b : 1-D numpy.array
    References
    ----------
    [1] Matlab R2016a `polystab` function 
    """
    if np.size(a) <= 1:
        return a
    ## Actual process
    v = np.roots(a)
    ii = np.where(v != 0)[0]
    vs = 0.5 * (np.sign(np.abs(v[ii]) - 1) + 1)
    v[ii] = (1 - vs) * v[ii] + vs / np.conj(v[ii])
    ind = np.where(a != 0)[0]
    b = a[ind[0]] * np.poly(v)
    ## Security
    if not (np.any(np.imag(a))):
        b = np.real(b)
    return b


def numf(h, a, nb):
    """
    Find numerator B given impulse-response h of B/A and denominator a
    Parameters
    ----------
    - h : real 1D array
            impulse-response.   
    - a : real 1D array
            denominator of the estimated filter.
    - nb : integer scalar
            numerator order.
        
    Returns
    -------
    - b : real 1D array.
            numerator of the estimated filter.
        
    References
    ----------
    [1] Matlab R2016a `yulewalk` function
    """
    nh = np.max(np.shape(h))
    impr = signal.lfilter([1.0], a, np.append(1.0, np.zeros(nh - 1)))
    b = np.matmul(h, linalg.pinv(linalg.toeplitz(impr, np.append(1.0, np.zeros(nb))).T))
    return b


def denf(R, na):
    """
    Compute filter denominator from covariances.
    A = denf(R,na) computes order na denominator A from covariances 
    R(0)...R(nr) using the Modified Yule-Walker method.  
    This function is used by yulewalk.
    Parameters
    ----------
    - R : real 1D array.
            Covariances.
        
    - na : integer scalar.
            Order of the denominator.
            
    Returns
    -------
    - A : real 1d array. 
            Denominator of the estimated filter.
    
    References
    ----------
    [1] Matlab R2016a `yulewalk` function
    """
    nr = np.max(np.shape(R))
    Rm = linalg.toeplitz(R[na:nr - 1], R[na:0:-1])
    Rhs = -R[na + 1:nr + 1]
    A = np.matmul(Rhs, linalg.pinv(Rm.T))
    return np.append([1], A)
