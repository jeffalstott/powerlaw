def neuro_band_filter(data, band, sampling_rate=1000.0, taps=513.0, window_type='blackmanharris', downsample=True):
    """docstring for neuro_band_filter"""
    from numpy import array, floor
    bands = {'delta': (array([4.0]), True),
            'theta': (array([4.0,8.0]), False),
            'alpha': (array([8.0,12.0]), False),
            'beta': (array([12.0,30.0]), False),
            'gamma': (array([30.0,80.0]), False),
            'high-gamma': (array([80.0, 100.0]), False),
            'broad': (array([100.0]), True),
            }
    if band=='raw':
        return data
    frequencies = bands[band][0]
    pass_zero= bands[band][1]
    from scipy.signal import firwin, lfilter
    nyquist = sampling_rate/2.0
    kernel= firwin(taps, frequencies/nyquist, pass_zero=pass_zero, window = window_type)
    data = lfilter(kernel, 1.0, data)
    if downsample==True or downsample=='nyquist':
        downsampling_interval = floor(( 1.0/ (2.0*frequencies.max()) )*sampling_rate)
    elif downsample:
        downsampling_interval = floor(sampling_rate/float(downsample))
    else:
        return data, frequencies, sampling_rate

    data = data[:,::downsampling_interval]
    downsampled_rate=sampling_rate/downsampling_interval
    return data, frequencies, downsampled_rate
