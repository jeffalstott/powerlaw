def neuro_band_filter(data, band, sampling_rate=1000.0):
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
    kernel= firwin(25.0, frequencies[0]/nyquist, pass_zero=frequencies[1])
    data = lfilter(kernel, 1.0, data)
    downsampling_rate = (1.0/(2.0*frequencies[0].max()))*sampling_rate
    data = data[:,:-1:downsampling_rate]
    return data

def avalanche_analysis(data,bin_width=1, percentile=.99, event_method='amplitude'):
    """docstring for avalanche_analysis  """
    raster_displacement, raster_amplitude, iei = find_events(data, percentile, event_method)
    starts, stops = find_cascades(raster_displacement, bin_width)

    metrics = {'iei_mean': iei.mean(),
            'start': starts,
            'stop': stops,
            'duration_silences': starts[1:]-stops[:-1],
            'bin_width': bin_width,
            'percentile': percentile,
            'event_method': event_method,
            }
    from numpy import concatenate, array

    #For every avalanche, calculate some list of metrics, then save those metrics in a dictionary
    #Metrics arrive as tuples of the form (label, array([value]))
    for i in range(len(starts)):
        m = avalanche_metrics(raster_displacement[:,starts[i]:stops[i]], \
                raster_amplitude[:,starts[i]:stops[i]], \
                bin_width)
        for k,v in m:
            metrics[k] = concatenate((metrics.setdefault(k,array([])), \
                    v))
    return metrics

def find_events(data_displacement, percentile=.99, event_method='amplitude', data_amplitude=0):
    """find_events does things"""
    from scipy.signal import hilbert
    from scipy.stats import scoreatpercentile
    from scipy.sparse import csc_matrix
    from numpy import rot90, tile, diff, sort, nonzero

    if ~data_amplitude:
        data_amplitude = abs(hilbert(data_displacement))
    
    if event_method == 'amplitude':
        signal = data_amplitude
    elif event_method == 'displacement':
        signal = data_displacement
    else:
        print 'Please select a valid event detection method (amplitude or displacement)'


    #scoreatpercentile only computes along the first dimension, which in this case is channels, not time. So we rotate the array before
    #we hand it to scoreatpercentile. The result is a vector of threshold values for each channel.
    threshold = scoreatpercentile(rot90(signal), percentile*100)

    #In order to avoid looping over each channel individually, we replicate the vector of thresholds to the number of time steps in the data. 
    #Then we rotate it to fit the shape of the data.
    threshold_matrix = rot90( tile( threshold, (len(data_displacement[0]), 1)), 3)
    raster_displacement = csc_matrix(data_displacement*\
            (signal>threshold_matrix))
    raster_amplitude = csc_matrix(data_amplitude*\
            (signal>threshold_matrix))
    interevent_intervals = diff(sort(nonzero(raster_displacement)[1]))
    return (raster_displacement, raster_amplitude, interevent_intervals)

def find_cascades(raster, bin_width=1):
    """find_events does things"""
    from numpy import concatenate, reshape, zeros, diff, size, nonzero, unique

    #Collapse the reaster into a vector of zeros and ones, indicating activity or inactivity on all channels
    z = zeros(raster.shape[1])
    activity = unique(nonzero(raster)[1])
    z[activity] = 1
    raster = z

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
    if size(stops)>0 and stops[-1]>data_points:
        stops[-1] = data_points

    return (starts, stops)

def avalanche_metrics(raster_displacement, raster_amplitude, bin_width):
    """avalanche_metrics calculates various things"""
    duration = raster_displacement.shape[1]

    from numpy import zeros, shape, nonzero
    raster_event = zeros(shape(raster_displacement))
    raster_event[nonzero(raster_displacement)] = 1
    raster_displacement = abs(raster_displacement)

    size_displacements = raster_displacement.sum()
    size_events = raster_event.sum()
    size_amplitudes = raster_amplitude.sum()

    if duration<(2*bin_width):
        sigma_amplitudes= 0
        sigma_events = 0
        sigma_displacements = 0
    else:
        sigma_displacements = raster_displacement[:,bin_width:(2*bin_width)].sum() \
                / raster_displacement[:,:bin_width].sum()

        sigma_amplitudes = raster_amplitude[:,bin_width:(2*bin_width)].sum() \
                / raster_amplitude[:,:bin_width].sum()

        sigma_events = raster_event[:,bin_width:(2*bin_width)].sum() \
                / raster_event[:,:bin_width].sum()

    from numpy import array
    metrics = [('duration', array([duration])), \
            ('size_amplitudes', array([size_amplitudes])),\
            ('size_displacements', array([size_displacements])),\
            ('sigma_amplitudes',array([sigma_amplitudes])),\
            ('sigma_displacements',array([sigma_displacements])),\
            ('size_events', array([size_events])), \
            ('sigma_events', array([sigma_events])), \
            ]
    return metrics

def log_hist(data, max_size, min_size=1, show=True):
    """log_hist does things"""
    from numpy import logspace, histogram
    from math import ceil, log10
    import pylab
    log_min_size = log10(min_size)
    log_max_size = log10(max_size)
    number_of_bins = ceil((log_max_size-log_min_size)*10)
    bins=logspace(log_min_size, log_max_size, num=number_of_bins)
    hist, edges = histogram(data, bins, density=True)
    if show:
        pylab.plot(edges[:-1], hist, 'o')
        pylab.gca().set_xscale("log")
        pylab.gca().set_yscale("log")
        pylab.show()
    return (hist, edges)
