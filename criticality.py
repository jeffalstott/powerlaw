def avalanche_analysis(data,bin_width=1, percentile=.01, event_method='amplitude'):
    """docstring for avalanche_analysis  """
    raster = find_events(data, percentile, event_method)
    indices = find_cascades(raster, bin_width)
    sizes = avalanche_statistics(data, indices)
    return sizes

def find_events(data, percentile=.01, event_method='amplitude'):
    """find_events does things"""
    from scipy.signal import hilbert
    from scipy.stats import scoreatpercentile
    from numpy import rot90, tile
    
    if event_method == 'amplitude':
        data = abs(hilbert(data))

    #scoreatpercentile only computes along the first dimension, which in this case is channels, not time. So we rotate the array before
    #we hand it to scoreatpercentile. The result is a vector of threshold values for each channel.
    threshold = scoreatpercentile(rot90(data), percentile*100)

    #In order to avoid looping over each channel individually, we replicate the vector of thresholds to the number of time steps in the data. 
    #Then we rotate it to fit the shape of the data.
    threshold_matrix = rot90( tile( threshold, (len(data[0]), 1)), 3)
    raster = data(data>threshold_matrix)
    return raster

def find_cascades(raster, bin_width=1):
    """find_events does things"""
    from numpy import concatenate, reshape, zeros

    #Collapse the reaster into a vector of zeros and ones, indicating activity or inactivity on all channels
    raster = raster>0
    raster = raster.sum(0)

    #Find how short we'll be trying to fill the last bin, then pad the end
    remainder = len(raster[0]) % bin_width
    raster = concatenate((raster, zeros(remainder)), 1)


    raster = reshape

    return indices

def avalanche_statistics(data, indices):
    """avalanche_statistics calculates various things"""
    return stats

