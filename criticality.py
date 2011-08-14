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

def avalanche_analysis(data,bin_width=1, percentile=.99, event_method='amplitude', data_amplitude=0):
    """docstring for avalanche_analysis  """
    metrics = {}
    metrics['bin_width'] = bin_width
    metrics['percentile'] = percentile
    metrics['event_method'] = event_method


    m = find_events(data, percentile, event_method, data_amplitude)
    metrics.update(m)

    from numpy import concatenate, array
    starts, stops = find_cascades(metrics['event_times'], bin_width)

    metrics['starts'] = starts
    metrics['stops'] = stops
    metrics['durations'] = stops-starts
    metrics['durations_silences'] = starts[1:]-stops[:-1]

    #For every avalanche, calculate some list of metrics, then save those metrics in a dictionary
    for i in range(len(starts)):
        m = avalanche_metrics(metrics, i)
        for k,v in m:
            metrics[k] = concatenate((metrics.setdefault(k,array([])), \
                    v))
    return metrics

def find_events(data_displacement, percentile=.99, event_method='amplitude', data_amplitude=0):
    """find_events does things"""
    from scipy.signal import hilbert
    from scipy.stats import scoreatpercentile
    import numpy

    if type(data_amplitude)!=numpy.ndarray:
        data_amplitude = abs(hilbert(data_displacement))

    if event_method == 'amplitude':
        signal = data_amplitude
    elif event_method == 'displacement':
        signal = data_displacement
    else:
        print 'Please select a supported event detection method (amplitude or displacement)'


    #scoreatpercentile only computes along the first dimension, so we transpose the 
    #(channels, times) matrix to a (times, channels) matrix. This is also useful for
    #applying the threshold, which is #channels long. We just need to make sure to 
    #invert back the coordinate system when we assign the results, which we do 
    threshold = scoreatpercentile(numpy.transpose(signal), percentile*100)
    times, channels = numpy.where(numpy.transpose(signal)>threshold)

    displacements = data_displacement[channels, times]
    amplitudes = data_amplitude[channels,times]
    interevent_intervals = numpy.diff(numpy.sort(times))

    output_metrics = { \
            'event_times': times, \
            'event_channels': channels,\
            'event_displacements': displacements,\
            'event_amplitudes': amplitudes,\
            'interevent_intervals': interevent_intervals,\
            }
    return output_metrics

def find_cascades(event_times, bin_width=1):
    """find_events does things"""
    from numpy import concatenate, reshape, zeros, diff, size, unique

    #Collapse the reaster into a vector of zeros and ones, indicating activity or inactivity on all channels
    raster = zeros(event_times.max()+1)
    raster[unique(event_times)] = 1

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

def avalanche_metrics(input_metrics, avalanche_number):
    """avalanche_metrics calculates various things"""
    from numpy import array, where
    avalanche_stop = where(input_metrics['event_times'] < \
            input_metrics['stops'][avalanche_number])[0][-1]+1
    avalanche_start = where(input_metrics['event_times'] >= \
            input_metrics['starts'][avalanche_number])[0][0]

    size_events = array([avalanche_stop-avalanche_start])
    size_displacements = array([\
            sum(abs(\
            input_metrics['event_displacements'][avalanche_start:avalanche_stop]))\
            ])
    size_amplitudes = array([\
            sum(abs(\
            input_metrics['event_amplitudes'][avalanche_start:avalanche_stop]))\
            ])

    if input_metrics['durations'][avalanche_number] < \
            (2*input_metrics['bin_width']):
                sigma_amplitudes = sigma_events = sigma_displacements = array([0])
    else:
        first_bin = where( \
                input_metrics['event_times'] < \
                (input_metrics['starts'][avalanche_number] \
                +input_metrics['bin_width'])\
                )[0][-1]
        second_bin = where( \
                input_metrics['event_times'] < \
                (input_metrics['starts'][avalanche_number] \
                +2*input_metrics['bin_width'])\
                )[0][-1]+1
        
        sigma_events = array([\
                (second_bin-first_bin)/ \
                (first_bin-avalanche_start+1.0) \
                ])
        sigma_displacements = array([\
                sum(abs(input_metrics['event_displacements'][first_bin:second_bin]))/  \
                sum(abs(input_metrics['event_displacements'][avalanche_start:first_bin+1]))\
                ])
        sigma_amplitudes = array([\
                sum(abs(input_metrics['event_amplitudes'][first_bin:second_bin]))/  \
                sum(abs(input_metrics['event_amplitudes'][avalanche_start:first_bin+1]))\
                ])

    output_metrics = (\
            ('size_amplitudes', size_amplitudes),\
            ('size_displacements', size_displacements),\
            ('sigma_amplitudes', sigma_amplitudes),\
            ('sigma_displacements', sigma_displacements),\
            ('size_events', size_events), \
            ('sigma_events', sigma_events), \
            )
    return output_metrics

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
