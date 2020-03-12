# Source: https://github.com/mrazavian/ITURPropagPY/blob/18eac5b66b95b868d23a93977872ae9fe6bbaac4/iturpropag/models/iturp1853/scintillation_attenuation_synthesis.py

# import numpy as np
# from scipy import signal, linalg

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
    lap = np.fix(npt/25)

    npt = npt + 1
    Ht = np.zeros(npt)

    nint = np.size(ff) - 1
    df = np.diff(ff)

    nb = 1
    Ht[0] = aa[0]
    for ii in np.arange(nint):

        if df[ii] == 0:
            nb = int(nb - lap/2)
            ne = int(nb + lap)
        else:
            ne = int(np.fix(ff[ii+1] * npt))
        
        jj = np.arange(nb, ne + 1)
        if ne == nb:
            inc = 0
        else:
            inc = (jj - nb) / (ne - nb); 
        
        Ht[nb -1 : ne] = inc * aa[ii + 1] + (1 - inc) * aa[ii]
        nb = int(ne + 1)
    
    Ht = np.append(Ht, Ht[-2:0:-1])
    n = np.size(Ht)
    n2 = int(np.fix((n+1) / 2))
    nb = na
    nr = 4 * na
    nt = np.arange(nr)

    R = np.real( np.fft.ifft(Ht * Ht) )
    R = R[:nr] * (0.54 + 0.46 * np.cos(np.pi * nt/(nr-1) ))

    Rwindow = np.append(0.5, np.ones(n2 - 1))
    Rwindow = np.append( Rwindow, np.zeros(n - n2) )

    A = self.polystab( self.denf(R, na) )

    R = R[:nr]
    R[0] = R[0]/2
    Qh = self.numf(R, A, na)

    _, Ss = 2* np.real(signal.freqz(Qh, A, n , whole=True))
    var1 = np.log( Ss.astype('complex') )
    var2 = np.fft.ifft(var1)
    hh = np.fft.ifft( np.exp( np.fft.fft(Rwindow * var2) ) )
    B = np.real( self.numf(hh[:nr], A, nb ))

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
    vs = 0.5 * (np.sign( np.abs( v[ ii ] ) - 1 ) + 1)
    v[ii] = (1 - vs) * v[ii] + vs / np.conj(v[ii])
    ind = np.where(a != 0)[0]
    b =  a[ ind[0] ] * np.poly(v)
    ## Security
    if not (np.any(np.imag(a))):
        b = np.real(b)
    return b

def numf(self, h, a, nb):
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
    b = np.matmul(h, linalg.pinv( linalg.toeplitz(impr, np.append(1.0, np.zeros(nb))).T ) )
    return b

def denf(self, R, na):
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