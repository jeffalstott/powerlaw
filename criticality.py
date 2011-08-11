def neuro_band_filter(data, band, sampling_rate=1000):
    """docstring for neuro_band_filter"""
    from numpy import array
    bands = {'delta': (array([4]), True),
            'theta': (array([4,8]), False),
            'alpha': (array([8,12]), False),
            'beta': (array([12,30]), False),
            'gamma': (array([30,80]), False),
            'high-gamma': (array([80]), False),
            'broad': (array([1,100]), False),
            }
    frequencies = bands[band]
    from scipy.signal import firwin, lfilter
    nyquist = sampling_rate/2
    kernel= firwin(25, frequencies[0]/nyquist, pass_zero=frequencies[1])
    data = lfilter(kernel, 1.0, data)
    return data





def avalanche_analysis(data,bin_width=1, percentile=.99, event_method='amplitude'):
    """docstring for avalanche_analysis  """
    raster = find_events(data, percentile, event_method)
    starts, stops = find_cascades(raster, bin_width)

    metrics = {'starts': starts, 'stops': stops}

    for i in range(len(starts)):
        m = avalanche_metrics(raster[:,starts[i]:stops[i]], bin_width)
        for k,v in m:
            metrics.setdefault(k,[]).append(v)

    return metrics

def find_events(data, percentile=.99, event_method='amplitude'):
    """find_events does things"""
    from scipy.signal import hilbert
    from scipy.stats import scoreatpercentile
    from numpy import rot90, tile
    
    if event_method == 'amplitude':
        signal = abs(hilbert(data))
    else:
        signal = data

    #scoreatpercentile only computes along the first dimension, which in this case is channels, not time. So we rotate the array before
    #we hand it to scoreatpercentile. The result is a vector of threshold values for each channel.
    threshold = scoreatpercentile(rot90(signal), percentile*100)

    #In order to avoid looping over each channel individually, we replicate the vector of thresholds to the number of time steps in the data. 
    #Then we rotate it to fit the shape of the data.
    threshold_matrix = rot90( tile( threshold, (len(data[0]), 1)), 3)
    raster = data*(data>threshold_matrix)
    return raster

def find_cascades(raster, bin_width=1):
    """find_events does things"""
    from numpy import concatenate, reshape, zeros, diff

    #Collapse the reaster into a vector of zeros and ones, indicating activity or inactivity on all channels
    raster = raster>0
    raster = raster.sum(0)
    raster = raster>0

    #Find how short we'll be trying to fill the last bin, then pad the end
    data_points = raster.shape[0]
    short = bin_width - (data_points % bin_width)
    raster = concatenate((raster, zeros(short)), 1)

    #Reshaped the raster vector into a bin_width*bins array, so that we can easily collapse bins together without using for loops
    raster = reshape( raster, (raster.shape[0]/bin_width, bin_width) )
    
    #Collapse bins together and find where the system switches states
    raster = raster.sum(1)
    raster = raster>0
    raster = diff(concatenate((zeros((1)), raster, zeros((1))), 1))
    #raster = squeeze(raster)

    starts = (raster==1).nonzero()[0]
    stops = (raster==-1).nonzero()[0]

    #Expand the bin indices back into the indices of the original raster
    starts = starts*bin_width
    stops = stops*bin_width
    #Additionally, if the data stops midway through a bin, and there is an avalanche in that bin, the above code will put the stop index in a later,
    #non-existent bin. Here we put the avalanche end at the end of the recording
    if stops[-1]>data_points:
        stops[-1] = data_points

    return (starts, stops)

def avalanche_metrics(raster, bin_width):
    """avalanche_metrics calculates various things"""
    duration = raster.shape[1]

    event_raster = raster>0

    size_amplitudes = raster.sum()
    size_events = event_raster.sum()

    if duration<(2*bin_width):
        sigma_amplitudes= 0
        sigma_events = 0
    else:
        sigma_amplitudes = raster[:,bin_width:(2*bin_width)].sum() \
                / raster[:,:bin_width].sum()

        sigma_events = event_raster[:,bin_width:(2*bin_width)].sum() \
                / event_raster[:,:bin_width].sum()


    metrics = [('duration', duration), ('size_amplitudes', size_amplitudes),\
            ('sigma_amplitudes',sigma_amplitudes), ('size_events', size_events), \
            ('sigma_events', sigma_events)]
    return metrics

