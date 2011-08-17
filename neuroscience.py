def neuro_band_filter(data, band, sampling_rate=1000.0, taps=511.0, window_type='blackmanharris'):
    """docstring for neuro_band_filter"""
    from numpy import array
    bands = {'delta': (array([4.0]), True),
            'theta': (array([4.0,8.0]), False),
            'alpha': (array([8.0,12.0]), False),
            'beta': (array([12.0,30.0]), False),
            'gamma': (array([30.0,80.0]), False),
            'high-gamma': (array([80.0]), False),
            'broad': (array([1.0,100.0]), False),
            }
    frequencies = bands[band]
    from scipy.signal import firwin, lfilter
    nyquist = sampling_rate/2.0
    kernel= firwin(taps, frequencies[0]/nyquist, pass_zero=frequencies[1], window = window_type)
    data = lfilter(kernel, 1.0, data)
    downsampling_rate = (1.0/(2.0*frequencies[0].max()))*sampling_rate
    data = data[:,:-1:downsampling_rate]
    return data, frequencies[0]
