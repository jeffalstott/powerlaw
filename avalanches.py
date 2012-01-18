from numpy import array, where, log2
from bisect import bisect_left
from sys import float_info

def avalanche_analysis(data,\
        data_amplitude=False, data_displacement_aucs=False, data_amplitude_aucs=False, \
        event_signal='displacement', event_detection='local_extrema',\
        threshold_mode='SD', threshold_level=3, threshold_direction='both',\
        time_scale=1, cascade_method='grid', \
        spatial_sample='all', spatial_sample_name=False,\
        temporal_sample='all', temporal_sample_name=False,\
        write_to_HDF5=False, overwrite_HDF5=False,\
        HDF5_group=None):
    """docstring for avalanche_analysis  """

    import h5py
#See if we've just been passed a reference to an HDF5 file. If so, load the relevant section.
    if type(data)==unicode or type(data)==str:
        data = h5py.File(data)[HDF5_group]

#If we don't have a name for the spatial or temporal samples, generate one
    if not spatial_sample_name:
        if spatial_sample=='all':
            spatial_sample_name='all'
        else:
            spatial_sample_name = str(len(spatial_sample))
    if not temporal_sample_name:
        if temporal_sample=='all':
            temporal_sample_name='all'
        else:
            temporal_sample_name = str(len(temporal_sample))

#If we're writing to HDF5, construct the version label we will save. 
#If that version already exists, and we're not overwriting it, then cancel the calculation
    if write_to_HDF5:
        version = 'ts-'+str(time_scale)+'_m-'+threshold_mode+'_t-'+str(threshold_level)[2:]+'_e-'+event_signal + '_ed-'+event_detection +'_c-'+ cascade_method +'_s-'+spatial_sample_name+'_t-'+temporal_sample_name
        if not overwrite_HDF5 and 'avalanches' in list(data) and version in list(data['avalanches']):
            return 'Calculation aborted for version '+version+' as results already exist and the option overwrite=False'

#If we're reading from a HDF5 file, load what data is available
    if type(data)==h5py._hl.group.Group:
        if 'displacement' in data:
            data_displacement = data['displacement'][:,:]
        else:
            raise IOError("'When using an HDF5 input, need a dataset called 'displacement'")
        if 'amplitude' in data:
            data_amplitude = data['amplitude'][:,:]
        if 'displacement_aucs' in data:
            data_displacement_aucs = data['displacement_aucs'][:,:]
        if 'amplitude_aucs' in data:
            data_amplitude_aucs = data['amplitude_aucs'][:,:]
    else:
        data_displacement = data

    n_rows, n_columns = data_displacement.shape

    from numpy import ndarray
    #If we don't have amplitude or area_under_the_curve information yet, calculate it
    if type(data_amplitude)!=ndarray:
        data_amplitude = fast_amplitude(data_displacement)
    if type(data_displacement_aucs)!=ndarray:
        data_displacement_aucs = area_under_the_curve(data_displacement)
    if type(data_amplitude_aucs)!=ndarray:
        data_amplitude_aucs = area_under_the_curve(data_amplitude)

    metrics = {}
    metrics['time_scale'] = time_scale
    metrics['threshold_mode'] = threshold_mode
    metrics['threshold_level'] = threshold_level
    metrics['event_signal'] = event_signal
    metrics['cascade_method'] = cascade_method
    metrics['spatial_sample'] = spatial_sample_name 
    metrics['temporal_sample'] = temporal_sample_name 

    if event_signal == 'amplitude':
        signal = data_amplitude
    elif event_signal == 'displacement':
        signal = data_displacement
    else:
        print 'Please select a supported event detection method (amplitude or displacement)'

    thresholds_up, thresholds_down = find_thresholds(signal, threshold_mode, threshold_level, threshold_direction)

    m = find_events(signal, thresholds_up=thresholds_up, thresholds_down=thresholds_down,\
        event_detection=event_detection, spatial_sample=spatial_sample, temporal_sample=temporal_sample)

    metrics.update(m)

    metrics['event_displacements'] = data_displacement[m['event_channels'], m['event_times']]
    metrics['event_amplitudes']= data_amplitude[m['event_channels'], m['event_times']]
    metrics['event_amplitude_aucs'] = data_amplitude_aucs[m['event_channels'], m['event_times']]
    metrics['event_displacement_aucs'] = data_displacement_aucs[m['event_channels'], m['event_times']]
    

    starts, stops = find_cascades(metrics['event_times'], time_scale, cascade_method)

    metrics['starts'] = starts
    metrics['stops'] = stops
    metrics['durations'] = (stops-starts).astype(float)
    metrics['durations_silences'] = (starts[1:]-stops[:-1]).astype(float)
    metrics['n'] = len(starts)

    #For every avalanche, calculate some list of metrics, then save those metrics in a dictionary
    from numpy import empty, ndarray
    n_avalanches = len(starts)
    n_events = len(metrics['event_times'])
    previous_event = 0
    for i in range(n_avalanches):
        m = avalanche_metrics(metrics, i)
        for k,v in m:
            if type(v)==ndarray:
                n_events_covered = max(v.shape)
                latest_event = previous_event+n_events_covered
                metrics.setdefault(k,empty((n_events)))[previous_event:latest_event] = v
            else:
                metrics.setdefault(k,empty((n_avalanches)))[i] = v
        previous_event = latest_event

    #Done calculating. Now to return or write to file
    if not(write_to_HDF5):
        return metrics
    else:
        #Assume we were given an HDF5 group in $data
        elements = list(data)
        if version in elements:
            print 'Avalanche analysis has already been done on these data with these parameters!'
            return metrics
        results_subgroup = data.create_group('avalanches/'+version)
        #Store parameters for this analysis (including some parameters formatted as strings)
        #as attributes of this version. All the numerical results we store as new datasets in
        #in this version group
        attributes = ('time_scale', 'threshold_mode', 'threshold_level', 'event_signal',\
                'cascade_method', 'spatial_sample', 'temporal_sample', 'n')
        for k in attributes:
            results_subgroup.attrs[k] = metrics[k]
        for k in metrics:
            if k not in attributes:
                if len(metrics[k])!=0:
                    results_subgroup.create_dataset(k, data=metrics[k])
                else:
                    results_subgroup.create_dataset(k, data=array([0]))
        return

def find_thresholds(signal, threshold_mode='SD', threshold_level=3, threshold_direction='both'):
    """find_thresholds does things"""
    from numpy import tile, shape, transpose, any

    s = shape(threshold_level)
    if len(signal.shape)==1:
        n_channels=1
    else:
        n_channels, n_time_points = signal.shape
    thresholds_down = None
    thresholds_up = None

    if threshold_mode == 'percentile':
        from scipy.stats import scoreatpercentile
        #scoreatpercentile only computes along the first dimension, so we transpose the 
        #(channels, times) matrix to a (times, channels) matrix, calculate the percentile value, then transpose back
        if s==() or s==(1,):
            if type(threshold_level)==array:
                threshold_level = threshold_level[0]
            if threshold_direction=='both':
                thresholds_up = transpose(scoreatpercentile(transpose(signal), threshold_level))
                thresholds_down = transpose(scoreatpercentile(transpose(signal), 100-threshold_level))
            elif threshold_direction=='up':
                thresholds_up = transpose(scoreatpercentile(transpose(signal), threshold_level))
            elif threshold_direction=='down':
                thresholds_down = transpose(scoreatpercentile(transpose(signal), threshold_level))
        elif s==(2,) and threshold_direction=='both':
            thresholds_up = transpose(scoreatpercentile(transpose(signal), threshold_level[0]))
            thresholds_down = transpose(scoreatpercentile(transpose(signal), threshold_level[1]))
        elif s[0]==n_channels and len(s)<2 and threshold_direction=='up':
             thresholds_up = [scoreatpercentile(signal[x,:],threshold_level[x]) for x in range(n_channels)]
        elif s[0]==n_channels and len(s)<2 and threshold_direction=='down':
             thresholds_down = [scoreatpercentile(signal[x,:],threshold_level[x]) for x in range(n_channels)]
        elif s[0]==n_channels and len(s)==2 and s[1]==2 and threshold_direction=='both':
             thresholds_up = [scoreatpercentile(signal[x,:],threshold_level[x,0]) for x in range(n_channels)]
             thresholds_down = [scoreatpercentile(signal[x,:],threshold_level[x,1]) for x in range(n_channels)]

    elif threshold_mode == 'SD':
        if s==() or s==(1,):
            if threshold_direction in ['both', 'up']:
                thresholds_up = signal.mean(-1) + signal.std(-1)*threshold_level
            if threshold_direction in ['both', 'down']:
                thresholds_down = signal.mean(-1) - signal.std(-1)*threshold_level
        elif s==(2,) and threshold_direction=='both':
            thresholds_up = signal.mean(-1) + signal.std(-1)*threshold_level[0]
            thresholds_down = signal.mean(-1) - signal.std(-1)*threshold_level[1]
        elif s[0]==n_channels and len(s)<2 and threshold_direction=='up':
            thresholds_up = signal.mean(-1) + signal.std(-1)*threshold_level
        elif s[0]==n_channels and len(s)<2 and threshold_direction=='down':
            thresholds_down = signal.mean(-1) - signal.std(-1)*threshold_level
        elif s[0]==n_channels and len(s)==2 and s[1]==2 and threshold_direction=='both':
            thresholds_up = signal.mean(-1) + signal.std(-1)*threshold_level[:,0]
            thresholds_down = signal.mean(-1) - signal.std(-1)*threshold_level[:,1]

    elif threshold_mode == 'Likelihood':
        from statistics import likelihood_threshold
        from numpy import zeros
        if threshold_direction in ['both', 'up']:
            thresholds_up = zeros(len(signal))
        if threshold_direction in ['both', 'down']:
            thresholds_down = zeros(len(signal))

        for i in range(len(signal)):
            if s==() or s==(1,):
                left_threshold, right_threshold = likelihood_threshold(signal[i], threshold_level)
                if threshold_direction in ['both', 'up']:
                    thresholds_up[i] = right_threshold
                if threshold_direction in ['both', 'down']:
                    thresholds_down[i] = left_threshold
            elif s==(2,) and threshold_direction=='both':
                left_threshold, right_threshold = likelihood_threshold(signal[i], threshold_level)
                thresholds_up[i] = right_threshold
                thresholds_down[i] = left_threshold
            elif s[0]==n_channels and len(s)<2:
                left_threshold, right_threshold = likelihood_threshold(signal[i], threshold_level[i])
                if threshold_direction in ['both', 'up']:
                    thresholds_up[i] = right_threshold
                if threshold_direction in ['both', 'down']:
                    thresholds_down[i] = left_threshold
            elif s[0]==n_channels and len(s)==2 and s[1]==2 and threshold_direction=='both':
                left_threshold, right_threshold = likelihood_threshold(signal[i], threshold_level[i])
                thresholds_up[i] = right_threshold
                thresholds_down[i] = left_threshold

    elif threshold_mode == 'absolute':
        s = shape(threshold_level)
        if s==() or s==(1,):
            if threshold_direction=='both': 
                thresholds_up = tile(abs(threshold_level), n_channels)
                thresholds_down = tile(-abs(threshold_level), n_channels)
            elif threshold_direction=='up':
                thresholds_up = tile(threshold_level, n_channels)
            elif threshold_direction=='down':
                thresholds_down = tile(threshold_level, n_channels)
        elif s==(2,) and threshold_direction=='both':
            thresholds_up = tile(threshold_level[0], n_channels)
            thresholds_down = tile(threshold_level[1], n_channels)
        elif s[0]==n_channels and len(s)<2 and threshold_direction=='up':
            thresholds_up = threshold_level
        elif s[0]==n_channels and len(s)<2 and threshold_direction=='down':
            thresholds_down = threshold_level
        elif s[0]==n_channels and len(s)==2 and s[1]==2 and threshold_direction=='both':
            thresholds_up = threshold_level[:,0]
            thresholds_down = threshold_level[:,1]

    if not any(thresholds_up) and not any(thresholds_down):
        raise IOError("Incoherent threshold levels array and/or threshold direction provided")

    return thresholds_up, thresholds_down


def find_events(signal, thresholds_up=None, thresholds_down=None,\
        event_detection='local_extrema',\
        spatial_sample='all', temporal_sample='all'):
    """find_events does things"""
    from numpy import diff, zeros, transpose, any, c_, r_

    if len(signal.shape)==1:
        n_channels=1
        n_time_points = signal.shape[0]
    else:
        n_channels, n_time_points = signal.shape

    #If we're not using all sensors or time points, take the samples now
    if spatial_sample=='all':
        spatial_sample=range(n_channels)
    spatial_mask = zeros((n_channels, n_time_points))
    spatial_mask[spatial_sample, :]=1

    if temporal_sample=='all':
        temporal_sample=range(n_time_points)
    temporal_mask = zeros((n_channels, n_time_points))
    temporal_mask[:,temporal_sample]=1

    sample_mask = spatial_mask*temporal_mask

    #Threshold mask
    up_mask = transpose(transpose(signal)>thresholds_up)*1
    down_mask = transpose(transpose(signal)<thresholds_down)*1

    if any(thresholds_up) and any(thresholds_down):
        threshold_mask = up_mask+down_mask
    elif any(thresholds_up):
        threshold_mask = up_mask
    elif any(thresholds_down):
        threshold_mask = down_mask

    #Event Definition
    if event_detection=='all':
        event_mask = threshold_mask
    else:
        if n_channels==1:
            max_mask = r_[zeros(1), signal[1:] > signal[ :-1]] * r_[signal[:-1] > signal[1:], zeros(1)]
            min_mask = r_[zeros(1), signal[1:] < signal[ :-1]] * r_[signal[:-1] < signal[1:], zeros(1)]
        else:
            max_mask = c_[zeros((n_channels,1)), signal[:,1:] > signal[:, :-1]] * c_[signal[:,:-1] > signal[:,1:], zeros((n_channels,1))]
            min_mask = c_[zeros((n_channels,1)), signal[:,1:] < signal[:, :-1]] * c_[signal[:,:-1] < signal[:,1:], zeros((n_channels,1))]

        if event_detection=='local':
            #all local minima/maxima
            event_mask = (max_mask + min_mask)*threshold_mask

        elif event_detection=='local_extrema':
            #local maxima for up, local minima for down
            event_mask = max_mask*up_mask + min_mask*down_mask

        elif event_detection=='excursion_extrema':
            #furthest up on any excursion above threshold_up, furthest down on any excursion below threshold_down
            up_event_mask = zeros(signal.shape)
            down_event_mask = zeros(signal.shape)
            for channel in range(n_channels):
                if any(thresholds_up):
                    if n_channels==1:
                        excursions = diff(up_mask)
                    else:
                        excursions = diff(up_mask[channel,:])
                    excursion_starts = where(excursions==1)[0]+1
                    excursion_stops = where(excursions==-1)[0]+1
                    if excursion_stops!=[] and excursion_stops[0]<excursion_starts[0]:
                        excursion_stops = excursion_stops[1:]
                    if len(excursion_stops)<len(excursion_starts):
                        excursion_starts = excursion_starts[0:-1]
                    for e in range(len(excursion_starts)):
                        if n_channels==1:
                            event_time = signal[excursion_starts[e]:excursion_stops[e]].argmax()
                            up_event_mask[event_time+excursion_starts[e]] = 1
                        else:
                            event_time = signal[channel, excursion_starts[e]:excursion_stops[e]].argmax()
                            up_event_mask[channel, event_time+excursion_starts[e]] = 1

                if any(thresholds_down): 
                    if n_channels==1:
                        excursions = diff(down_mask)
                    else:
                        excursions = diff(down_mask[channel,:])
                    excursion_starts = where(excursions==1)[0]+1
                    excursion_stops = where(excursions==-1)[0]+1
                    if excursion_stops!=[] and excursion_stops[0]<excursion_starts[0]:
                        excursion_stops = excursion_stops[1:]
                    if len(excursion_stops)<len(excursion_starts):
                        excursion_starts = excursion_starts[0:-1]
                    for e in range(len(excursion_starts)):
                        if n_channels==1:
                            event_time = signal[excursion_starts[e]:excursion_stops[e]].argmax()
                            down_event_mask[event_time+excursion_starts[e]] = 1
                        else:
                            event_time = signal[channel, excursion_starts[e]:excursion_stops[e]].argmax()
                            down_event_mask[channel, event_time+excursion_starts[e]] = 1

            event_mask = down_event_mask+up_event_mask

    event_matrix = event_mask*sample_mask
    #Find everywhere there is an event within the spatiotemporal sample. We want to sort these events by time
    #(the column axis), and the "where" function sorts by the row axis, so we transpose the event_matrix before
    #sending it to where. We just have to make sure to assign times and channels appropriately
    if len(signal.shape)==1:
        from numpy import zeros
        times = where( transpose(event_matrix) )[0]
        channels = zeros(len(times))
    else:
        times, channels = where( transpose(event_matrix) )

    #If we're not using all time points, take the temporal sample now
#    if temporal_sample!='all':
#        allowed_times = []
#        for i in len(times):
#            q = bisect_left(temporal_sample, times[i])
#            if q!=len(temporal_sample) and temporal_sample[q]==times[i]:
#                allowed_times.append(i)
#        times = times[allowed_times]
#        channels = channels[allowed_times]

    interevent_intervals = diff(times)

    output_metrics = { \
            'event_times': times, \
            'event_channels': channels,\
            'interevent_intervals': interevent_intervals,\
            }
    return output_metrics

def find_cascades(event_times, time_scale=1, method='grid'):
    """find_events does things"""
    from numpy import diff, concatenate

    if method=='gap':
        starts = array([event_times[0]])
        stops = array([event_times[-1]])
        changes = where(diff(event_times)>=time_scale+1)[0]
        starts = concatenate((starts, event_times[changes+1]))
        stops = concatenate((event_times[changes], stops))

    elif method=='grid':
        bin_width=time_scale
        from numpy import reshape, zeros, size, unique
        
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

    else:
        print 'Please select a supported cascade detection method (grid or gap)'

    return (starts, stops)

def avalanche_metrics(input_metrics, avalanche_number):
    """avalanche_metrics calculates various things"""
    #Index of leftmost item in x greater than or equal to y
    avalanche_stop = bisect_left(input_metrics['event_times'], \
            input_metrics['stops'][avalanche_number])
    avalanche_start = bisect_left(input_metrics['event_times'], \
            input_metrics['starts'][avalanche_number])

#Calculate sizes
    size_events = float(avalanche_stop-avalanche_start)
    size_displacements = sum(abs(\
            input_metrics['event_displacements'][avalanche_start:avalanche_stop]))
    size_amplitudes = sum(abs(\
            input_metrics['event_amplitudes'][avalanche_start:avalanche_stop]))
    size_displacement_aucs = sum(abs(\
            input_metrics['event_displacement_aucs'][avalanche_start:avalanche_stop]))
    size_amplitude_aucs = sum(abs(\
            input_metrics['event_amplitude_aucs'][avalanche_start:avalanche_stop]))
#Calculate sigmas
    if input_metrics['durations'][avalanche_number] < \
            (2*input_metrics['time_scale']):
                sigma_amplitudes = sigma_events = \
                        sigma_displacements = sigma_amplitude_aucs = \
                        0
    else:
        first_bin = bisect_left( \
                input_metrics['event_times'], \
                (input_metrics['starts'][avalanche_number] \
                +input_metrics['time_scale'])\
                )-1
        second_bin = bisect_left( \
                input_metrics['event_times'], \
                (input_metrics['starts'][avalanche_number] \
                +2*input_metrics['time_scale'])\
                )
        
        sigma_events = (second_bin-first_bin)/ \
                (first_bin-avalanche_start+1.0)
        sigma_displacements = \
                sum(abs(input_metrics['event_displacements'][first_bin:second_bin]))/  \
                sum(abs(input_metrics['event_displacements'][avalanche_start:first_bin+1]))
        sigma_amplitudes = \
                sum(abs(input_metrics['event_amplitudes'][first_bin:second_bin]))/  \
                sum(abs(input_metrics['event_amplitudes'][avalanche_start:first_bin+1]))
        sigma_amplitude_aucs = \
                sum(abs(input_metrics['event_amplitude_aucs'][first_bin:second_bin]))/  \
                sum(abs(input_metrics['event_amplitude_aucs'][avalanche_start:first_bin+1]))

#Calculate Tara's growth ratio
    event_times_within_avalanche = (\
            input_metrics['event_times'][avalanche_start:avalanche_stop] - \
            input_metrics['event_times'][avalanche_start]
            )

    initial_amplitude = \
            input_metrics['event_amplitudes'][avalanche_start].sum()
    t_ratio_amplitude = log2(\
            input_metrics['event_amplitudes'][avalanche_start:avalanche_stop] / \
            initial_amplitude \
            )

    initial_displacement = \
            abs(input_metrics['event_displacements'][avalanche_start]).sum()
    t_ratio_displacement = log2(\
            abs(input_metrics['event_displacements'][avalanche_start:avalanche_stop]) / \
            initial_displacement \
            )

    initial_amplitude_auc = \
            input_metrics['event_amplitude_aucs'][avalanche_start].sum()
    t_ratio_amplitude_auc = log2(\
            input_metrics['event_amplitude_aucs'][avalanche_start:avalanche_stop] / \
            initial_amplitude_auc \
            )

    initial_displacement_auc = \
            abs(input_metrics['event_displacement_aucs'][avalanche_start]).sum()
    t_ratio_displacement_auc = log2(\
            abs(input_metrics['event_displacement_aucs'][avalanche_start:avalanche_stop]) / \
            initial_displacement_auc \
            )
    output_metrics = (\
            ('size_events', size_events), \
            ('size_displacements', size_displacements),\
            ('size_amplitudes', size_amplitudes),\
            ('size_displacement_aucs', size_displacement_aucs), \
            ('size_amplitude_aucs', size_amplitude_aucs), \
            ('sigma_events', sigma_events), 
            ('sigma_displacements', sigma_displacements),\
            ('sigma_amplitudes', sigma_amplitudes),\
            ('sigma_amplitude_aucs', sigma_amplitude_aucs),\
            ('event_times_within_avalanche', event_times_within_avalanche), \
            ('t_ratio_displacements', t_ratio_displacement),\
            ('t_ratio_amplitudes', t_ratio_amplitude),\
            ('t_ratio_displacement_aucs', t_ratio_displacement_auc),
            ('t_ratio_amplitude_aucs', t_ratio_amplitude_auc),
            )
    return output_metrics

def area_under_the_curve(data, baseline='mean'):
    """area_under_the_curve is currently a mentally messy but computationally fast way to get an array of area under the curve information, to be used to assign to events. The area under the curve is the integral of the deflection from baseline (mean signal) in which an event occurrs. area_under_the_curve returns an array of the same size as the input data, where the datapoints are the areas of the curves the datapoints are contained in. So, all values located within curve N are the area of curve N, all values located within curve N+1 are the area of curve N+1, etc. Note that many curves go below baseline, so negative areas can be returned."""
    from numpy import cumsum, concatenate, zeros, empty, shape, repeat, diff, where, sign, ndarray
    n_rows, n_columns = data.shape

    if baseline=='mean':
        baseline = data.mean(1).reshape(n_rows,1)
    elif type(baseline)!=ndarray:
        print 'Please select a supported baseline_method (Currently only support mean and an explicit array)'

    #Convert the signal to curves around baseline
    curves_around_baseline = data-baseline

    #Take the cumulative sum of the signals. This will be rising during up curves and decreasing during down curves
    sums = cumsum(curves_around_baseline, axis=-1)
    #Find where the curves are, then where they stop
    z = zeros((n_rows,1))
    sums_to_diff = concatenate((z, sums, z), axis=-1)
    curving = sign(diff(sums_to_diff)) #1 during up curve and -1 during down curve
    curve_changes = diff(curving) #-2 at end of up curve and 2 at end of down curve
    curve_changes[:,-1] = 2 # Sets the last time point to be the end of a curve
    stop_channels, stop_times =where(abs(curve_changes)==2)
    stop_times = stop_times.clip(0,n_columns-1) #corrects for a +1 offset that can occur in a curve that ends at the end of the recording (in order to detect it we add an empty column at the end of the time series, but that puts the "end" of the curve 1 step after the end of the time series)

    data_aucs = empty(shape(data))
    for i in range(n_rows):
    #The value in the cumulative sum at a curve's finish will be the sum of all curves so far. So the value of the most recently finished curve is just the cumsum at this curve minus the cumsum at the end of the previous curve
        curves_in_row = where(stop_channels==i)[0]
        stops_in_row = stop_times[curves_in_row]
        if stops_in_row[0]==1: #If the first stop occurs at index 1, that means there's a curve at index 0 of duration 1
            stops_in_row = concatenate(([0],stops_in_row))
        values = sums[i,stops_in_row]-concatenate(([0],sums[i,stops_in_row[:-1]]))
        previous_stops = concatenate(([-1], stops_in_row[:-1]))
        durations = stops_in_row-previous_stops
        data_aucs[i] = repeat(values, durations)

    return data_aucs

def avalanche_statistics(metrics, \
        given_xmin_xmax=[(None,None)],\
        session=None, database_url=None, overwrite_database=False, \
        analysis_id=None, filter_id=None, \
        subject_id=None, task_id=None, experiment_id=None, sensor_id=None, recording_id=None,\
        close_session_at_end=False):
    from scipy.stats import mode, linregress
    from numpy import empty, unique, median, sqrt, sort
    import statistics as pl_statistics
    

    if not session and database_url:
        from sqlalchemy import create_engine
        from sqlalchemy.orm.session import Session
        engine = create_engine(database_url, echo=False)
        session = Session(engine)
        close_session_at_end=True

    if session:
        import database_classes as db
        avalanche_analysis = session.query(db.Avalanche).filter_by(\
                id=analysis_id).first()

	#If there are statistics already calculated, and we're not overwriting, then end.
        if avalanche_analysis.fits and not overwrite_database:
            return
        #Throw away the connection until we need it again, which will be awhile
        session.close()
        session.bind.dispose()

    #Various variable initializations for later
    statistics = {}
    times_within_avalanche = unique(metrics['event_times_within_avalanche'])
    j = empty(times_within_avalanche.shape)

    number_of_channels = len(unique(metrics['event_channels']))
    given_xmin_xmax =[(xmin, number_of_channels) if xmax=='channels' else (xmin, xmax) for xmin, xmax in given_xmin_xmax]

    #This list must start with 'power_law'!
    distributions_to_fit = [('power_law','alpha', 'error', None), ('truncated_power_law', 'alpha', 'gamma', None), ('exponential', 'gamma', None, None), ('lognormal', 'mu', 'sigma', None)] 

    for k in metrics:
        if k.startswith('sigma'):
            statistics[k]=metrics[k].mean()

            if session:
                setattr(avalanche_analysis, k, statistics[k])

        elif k.startswith('interevent_intervals'):
            statistics[k+'_mean']=metrics[k].mean()
            statistics[k+'_median']=median(metrics[k])
            statistics[k+'_mode']=mode(metrics[k])[0][0]

            if session:
                setattr(avalanche_analysis, k+'_mean', statistics[k+'_mean'])
                setattr(avalanche_analysis, k+'_median', statistics[k+'_median'])
                setattr(avalanche_analysis, k+'_mode', statistics[k+'_mode'])
                
        elif k.startswith('t_ratio'):
            statistics[k] = {}
            for i in range(len(times_within_avalanche)):
                j[i] = mode(metrics[k][metrics['event_times_within_avalanche']==times_within_avalanche[i]])[0][0] 
            regress = linregress(times_within_avalanche, j)
            statistics[k]['slope'] = regress[0]
            statistics[k]['R'] = regress[2]
            statistics[k]['p'] = regress[3]

            if session:
                setattr(avalanche_analysis, k+'_slope', statistics[k]['slope'])
                setattr(avalanche_analysis, k+'_R', statistics[k]['R'])
                setattr(avalanche_analysis, k+'_p', statistics[k]['p'])

        elif k.startswith('duration') or k.startswith('size'):
            statistics[k]={}
            if k.startswith('duration'):
                discrete=True
                xmin_xmax = [(None,None)]
            elif k.startswith('size_events'):
                discrete=True
                xmin_xmax = given_xmin_xmax
            else:
                discrete=False
                xmin_xmax = [(None,None)]

            for xmin, xmax in xmin_xmax:
                fixed_xmin=bool(xmin)
                fixed_xmax=bool(xmax)
                for distribution, parameter0, parameter1, parameter2 in distributions_to_fit:

                    if distribution=='power_law':
                        if xmin:
                            parameters, loglikelihood = pl_statistics.distribution_fit(\
                                    metrics[k], distribution, discrete=discrete, xmin=xmin, xmax=xmax)
                            alpha = parameters[0]
                            D = pl_statistics.power_law_ks_distance(sort(metrics[k]),\
                                    alpha, xmin=xmin, xmax=xmax, discrete=discrete)

                            D_plus_critical_branching, D_minus_critical_branching, Kappa = pl_statistics.power_law_ks_distance(sort(metrics[k]),\
                                    1.5, xmin=xmin, xmax=xmax, discrete=discrete, kuiper=True)
                            noise_flag = None
                            n_tail = sum(metrics[k]>=xmin)
                            alpha_error = (alpha-1)/sqrt(n_tail)
                            parameters = [alpha, alpha_error]

                            R=None
                            p=None
                        else:
                            xmin, D, alpha, loglikelihood, n_tail, noise_flag = pl_statistics.find_xmin(metrics[k],discrete=discrete)
                            D_plus_critical_branching, D_minus_critical_branching, Kappa = pl_statistics.power_law_ks_distance(sort(metrics[k]),\
                                    1.5, xmin=xmin, xmax=xmax, discrete=discrete, kuiper=True)
                            alpha_error = (alpha-1)/sqrt(n_tail)
                            parameters = [alpha, alpha_error]
                            R = None
                            p = None
                   # elif not xmax and discrete and (distribution=='lognormal' or distribution=='truncated_power_law'):
                   #     parameters, loglikelihood, R, p = pl_statistics.distribution_fit(metrics[k], distribution,\
                   #             xmin=xmin, xmax=max(metrics[k]), discrete=discrete, comparison_alpha=alpha)
                   #     D=None
                   #     D_plus_critical_branching=None
                   #     D_minus_critical_branching=None
                   #     Kappa = None
                    else:
                        parameters, loglikelihood, R, p = pl_statistics.distribution_fit(metrics[k], distribution,\
                                xmin=xmin, xmax=xmax, discrete=discrete, comparison_alpha=alpha)
                        D=None
                        D_plus_critical_branching=None
                        D_minus_critical_branching=None
                        Kappa = None

                    statistics[k][distribution]={}
                    statistics[k][distribution]['parameter1_name']=parameter0
                    statistics[k][distribution]['parameter1_value']=parameters[0]

                    statistics[k][distribution]['parameter2_name']=parameter1
                    if parameter1:
                        statistics[k][distribution]['parameter2_value']=parameters[1]
                    else:
                        statistics[k][distribution]['parameter2_value']=None

                    statistics[k][distribution]['parameter3_name']=parameter2
                    if parameter2:
                        statistics[k][distribution]['parameter3_value']=parameters[2]
                    else:
                        statistics[k][distribution]['parameter3_value']=None

                    statistics[k][distribution]['fixed_xmin']=fixed_xmin
                    statistics[k][distribution]['xmin']=xmin
                    statistics[k][distribution]['fixed_xmax']=fixed_xmax
                    statistics[k][distribution]['xmax']=xmax
                    statistics[k][distribution]['loglikelihood']=loglikelihood
                    statistics[k][distribution]['loglikelihood_ratio']=R
                    statistics[k][distribution]['p']=p
                    statistics[k][distribution]['KS']= D
                    statistics[k][distribution]['D_plus_critical_branching']= D_plus_critical_branching
                    statistics[k][distribution]['D_minus_critical_branching']= D_minus_critical_branching
                    statistics[k][distribution]['Kappa']= Kappa
                    statistics[k][distribution]['noise_flag']= noise_flag
                    statistics[k][distribution]['n_tail']=n_tail
                    statistics[k][distribution]['discrete']=discrete

                    if session:
                        fit_variables = statistics[k][distribution].keys()
                        distribution_fit = db.Fit(analysis_type='avalanches',\
                                variable=k, distribution=distribution,\
                                subject_id=subject_id, task_id=task_id, experiment_id=experiment_id,\
                                sensor_id=sensor_id, recording_id=recording_id, filter_id=filter_id,\
                                analysis_id=analysis_id)

                        for variable in fit_variables:
                            if statistics[k][distribution][variable]==float('inf'):
                                setattr(distribution_fit,variable, 1*10**float_info.max_10_exp)
                            elif statistics[k][distribution][variable]==-float('inf'):
                                setattr(distribution_fit,variable, -1*10**float_info.max_10_exp)
                            else:
                                setattr(distribution_fit,variable, statistics[k][distribution][variable])

                        avalanche_analysis.fits.append(distribution_fit)

    if session:
        session.add(avalanche_analysis)
        session.commit()
        if close_session_at_end:
            session.close()
            session.bind.dispose()
    return statistics

def avalanche_analyses(data,\
        threshold_mode, threshold_levels, threshold_directions,\
        event_signals, event_detections,\
        time_scales, cascade_methods,\
        spatial_samples=('all','all'), temporal_samples=('all','all'), \
        given_xmin_xmax=[(None, None)],\
        spatial_sample_names=None, temporal_sample_names=None, \
        write_to_HDF5=False, overwrite_HDF5=False,\
        HDF5_group=None,\
        session=None, database_url=None, overwrite_database=False,\
        filter_id=None, subject_id=None, task_id=None, experiment_id=None, sensor_id=None, recording_id=None,\
        data_amplitude=None, data_displacement_aucs=None, data_amplitude_aucs=None,\
        cluster=False, swarms_directory=None, analyses_directory=None, python_location=None,\
        close_session_at_end=False, verbose=False):

    if spatial_sample_names:
        spatial_samples = zip(spatial_samples, spatial_sample_names)
    elif type(spatial_samples[0])!=tuple:
        print 'Requires a list of spatial_samples AND a list of spatial_sample names, either as spatial_sample_names=list or as spatial_samples=a zipped list of tuples with indices and labels'
        return
    if temporal_sample_names:
        temporal_samples = zip(temporal_samples, temporal_sample_names)
    elif type(temporal_samples[0])!=tuple:
        print 'Requires a list of temporal_samples AND a list of temporal_sample names, either as temporal_sample_names=list or as temporal_samples=a zipped list of tuples with indices and labels'
        return
    analysis_id=None 
    if verbose:
        results = {}

    if not session and database_url:
        from sqlalchemy import create_engine
        from sqlalchemy.orm.session import Session
        engine = create_engine(database_url, echo=False)
        session = Session(engine)
    if session:
        import database_classes as db
    from sqlalchemy import and_

    parameter_space = [(tl, td, e, ed, ts, c,s,sn,t,tn) for tl in threshold_levels \
            for td in threshold_directions \
            for e in event_signals for ed in event_detections \
            for ts in time_scales for c in cascade_methods \
            for s,sn in spatial_samples \
            for t,tn in temporal_samples]

    if cluster:
        try:
            max_swarm = int(open(swarms_directory+'max_swarm_file.txt', 'r').read())
        except:
            print("Constructing max_swarm_file")
            from os import listdir
            swarms = [int(a) for a in listdir(swarms_directory)]
            if swarms:
                max_swarm = max(swarms)
            else:
                max_swarm = 0
        new_swarm = str(max_swarm+1)
        swarm_file = open(swarms_directory+new_swarm, 'w')
        analyses_number = 0

        try:
            max_analysis = int(open(analyses_directory+'max_analysis_file.txt', 'r').read())
        except:
            print("Constructing max_analysis_file")
            from os import listdir
            analyses = [int(a[:-3]) for a in listdir(analyses_directory)]
            if analyses:
                max_analysis = max(analyses)
            else:
                max_analysis = 0

    for tl, td, e, ed, ts, c,s,sn,t,tn in parameter_space:
        parameters = str(ts)+'_'+str(tl)+'_'+td+'_'+str(e)+'_'+ed+'_'+str(c)+'_'+str(sn)+'_'+str(tn)
        if verbose:
            results[parameters] = {}
            print parameters

        if session:
            threshold_tolerance = .000001*tl 
            #This is a hack to deal with storing all numbers as floats. Your database interface may (as mine does) result in switching back and forth between float32 and float64, which makes direction threshold_level=tl identification impossible.
            analysis = session.query(db.Avalanche).filter_by(\
                    filter_id=filter_id, spatial_sample=sn, temporal_sample=tn,\
                    threshold_mode=threshold_mode, threshold_direction=td,\
                    time_scale=ts, event_signal=e, event_detection=ed, cascade_method=c).\
                    filter(\
                            and_(db.Avalanche.threshold_level>(tl-threshold_tolerance),\
                            db.Avalanche.threshold_level<(tl+threshold_tolerance))).first()

            #If we're not overwriting the database, and there is a previous analysis with saved statistics, then go on to the next set of parameters
            if analysis:
                print("This analysis was previously started!")
            if not overwrite_database and analysis and analysis.fits:
                print("This analysis was already done. Skipping.")
                continue

            if analysis:
                analysis_id = analysis.id

        if not cluster:
            if session:
                session.commit()
                session.close()
                session.bind.dispose()
                #Throw away the connection until we need it again, which could be awhile

            metrics = avalanche_analysis(data,\
                    data_amplitude=data_amplitude, \
                    data_displacement_aucs=data_displacement_aucs,\
                    data_amplitude_aucs=data_amplitude_aucs,\
                    threshold_mode=threshold_mode, threshold_level=tl, threshold_direction=td,\
                    event_signal=e, event_detection=ed,\
                    time_scale=ts, cascade_method=c,\
                    spatial_sample=s, spatial_sample_name=sn,\
                    temporal_sample=t, temporal_sample_name=tn,\
                    write_to_HDF5=write_to_HDF5, overwrite_HDF5=overwrite_HDF5,\
                    HDF5_group=HDF5_group)


            if session and not analysis: 
                analysis = db.Avalanche(\
                        filter_id=filter_id, spatial_sample=sn, temporal_sample=tn,\
                        threshold_mode=threshold_mode, threshold_level=tl, threshold_direction=td,\
                        event_signal=e, event_detection=ed,\
                        time_scale=ts, cascade_method=c,\
                        subject_id=subject_id, task_id=task_id, experiment_id=experiment_id,\
                        sensor_id=sensor_id, recording_id=recording_id,\
                        n=metrics['n'],\
                        fits = [])
                session.add(analysis)
                session.commit()
                analysis_id=analysis.id

            statistics = avalanche_statistics(metrics, \
                    given_xmin_xmax=given_xmin_xmax,\
                    session=session, database_url=database_url, \
                    subject_id=subject_id, task_id=task_id, experiment_id=experiment_id,\
                    sensor_id=sensor_id, recording_id=recording_id, \
                    filter_id=filter_id, analysis_id=analysis_id)
            if verbose:
                results[parameters]['metrics'] = metrics 
                results[parameters]['statistics'] = statistics

        else:
            new_analysis = str(max_analysis+1)
            max_analysis = max_analysis+1
            analysis_file = open(analyses_directory+new_analysis+'.py', 'w')
            print("Writing analysis file "+new_analysis)

            analysis_file.write("database_url= %r\n\n" % database_url)
            analysis_file.write("session=False\n\n")

            analysis_file.writelines(['from avalanches import avalanche_analysis, avalanche_statistics\n',
                'from sqlalchemy import create_engine\n',
                'from sqlalchemy.orm.session import Session\n',
                'engine = create_engine(database_url, echo=False)\n\n'])

            analysis_file.write('analysis_id=%s \n\n' % analysis_id)
            analysis_file.write("""print("analysis_id=%r, threshold_mode=%r, threshold_level=%r, threshold_direction=%r, event_signal=%r, event_detection=%r, time_scale=%s, cascade_method=%r, spatial_sample=%r, spatial_sample_name=%r, temporal_sample=%r, temporal_sample_name=%r") \n\n"""\
                % (analysis_id, threshold_mode, tl, td, e, ed, ts, c, s,sn, t, tn))

            analysis_file.writelines(["metrics = avalanche_analysis(%r,\\\n" % data,
                "    threshold_mode=%r, threshold_level=%r, threshold_direction=%r,\\\n" % (threshold_mode,tl, td),
                "    event_signal=%r, event_detection=%r,\\\n" % (e,ed),
                "    time_scale=%s, cascade_method=%r,\\\n" % (ts,c),
                "    spatial_sample=%r, spatial_sample_name=%r,\\\n" % (s,sn),
                "    temporal_sample=%r, temporal_sample_name=%r,\\\n" % (t,tn),
                "    write_to_HDF5=%r, overwrite_HDF5=%r,\\\n" % (write_to_HDF5, overwrite_HDF5),
                "    HDF5_group=%r)\n\n" % HDF5_group])

            analysis_file.writelines(["if not analysis_id:\n",
                "    import database_classes as db\n",
                "    analysis = db.Avalanche(\\\n",
                "        filter_id=%s,\\\n" % filter_id,
                "        spatial_sample=%r, temporal_sample=%r,\\\n" % (sn, tn),
                "        threshold_mode=%r, threshold_level=%r, threshold_direction=%r,\\\n" % (threshold_mode, tl,td),
                "        event_signal=%r, event_detection=%r,\\\n" % (e, ed),
                "        time_scale=%r, cascade_method=%r,\\\n" % (ts, c),
                "        subject_id=%s, task_id=%s, experiment_id=%s,\\\n" % (subject_id, task_id, experiment_id),
                "        sensor_id=%s, recording_id=%s,\\\n" % (sensor_id, recording_id),
                "        n=metrics['n'],\\\n",
                "        fits = [])\n\n",
                "    session = Session(engine)\n",
                "    session.add(analysis)\n",
                "    session.commit()\n",
                "    analysis_id=analysis.id\n\n"])

            analysis_file.writelines(["statistics = avalanche_statistics(metrics,\\\n",
                "    given_xmin_xmax=%r,\\\n" % given_xmin_xmax,
                "    session=session, database_url=database_url,\\\n",
                "    subject_id=%s, task_id=%s, experiment_id=%s,\\\n" % (subject_id, task_id, experiment_id),
                "    sensor_id=%s, recording_id=%s,\\\n" % (sensor_id, recording_id),
                "    filter_id=%s, analysis_id=%s)\n\n" % (filter_id, analysis_id)])

            analysis_file.writelines(["if session:\n",
                "    session.close()\n",
                "    session.bind.dispose()\n"])

            analysis_file.close()

            swarm_file.write(python_location+' '+analyses_directory+new_analysis+'.py'+\
                    ' 2>&1  > '+analyses_directory+new_analysis+'_out\n')
            analyses_number = analyses_number+1
            

    if cluster:
        swarm_file.close()
        if analyses_number==0:
            print("No new analyses, not submitting swarm file "+new_swarm)
        else:
            from os import system
            print("Submitting "+str(analyses_number)+" analyses with swarm file "+new_swarm)
            system('swarm -f '+swarms_directory+new_swarm+' -g 8 -m a')
            open(swarms_directory+'max_swarm_file.txt', 'w').write(str(max_swarm+1))
            open(analyses_directory+'max_analysis_file.txt', 'w').write(str(max_analysis))

    if session:
        session.commit()
        if close_session_at_end:
            session.close()
            session.bind.dispose()

    if verbose:
        return results
    return

def energy_levels(data, time_scales):
    """energy_levels does things"""
    from numpy import ndarray, concatenate, zeros
    if type(time_scales)==list:
        time_scales = array(time_scales)
    if not(type(time_scales)==ndarray or type(time_scales)==array):
        from numpy import array
        time_scales = array([time_scales])
    if time_scales[0] == 0:
        time_scales +=1

    levels = {}

    n_columns = data.shape[-1]
    for i in time_scales:
        d = concatenate( (sum(data, 0), \
                zeros(i-n_columns%i)))

        windows = d.shape[0]/i
        x = zeros(windows)
        for j in range(i):
            x += d[j::i]
        levels[i] = x[:-1]

    return levels

def fast_amplitude(data):
    """Uses Scipy's hilbert function to calculate the amplitude envelope of a signal.\
            Importantly, Scipy's implementation can be very slow, depending on the factors \
            of the length of the signal. fast_amplitude pads the signal (taking more memory) \
            to the next factor of two to make the factorization fast!"""
    from scipy.signal import hilbert
    from numpy import zeros, concatenate
    n_rows, n_columns = data.shape
    target = next_power_of_2(n_columns) #Pad the array with zeros to the next power of 2 to speed up the Hilbert transform, which recursively calls DFT
    shortage = target-n_columns
    hd = abs(hilbert( \
            concatenate((data, zeros((n_rows, shortage))), axis=-1)))
    return hd[:,:n_columns]

def next_power_of_2(x):
    x -= 1
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16
    x += 1
    return x 

def tgrowth_graph(data, measure='amplitudes', time_steps=50, x_min=-2, x_max=2):

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import colorConverter
    from numpy import unique, histogram, where, arange
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')

    if measure[-1]!='s':
        measure = measure+'s'

    verts = []
    times = unique(data['event_times_within_avalanche'])
    valid_times = []
    prob_mean = []
    prob_std = []

    for z in arange(time_steps)+1:
        if z not in times:
            continue
        d = data['t_ratio_'+measure][where(data['event_times_within_avalanche']==z)[0]]
        probs, bins = histogram(d, normed=True)
        verts.append([(bins[0], 0)] + zip(bins[:-1], probs) + [(bins[-2],0)])
        valid_times.append(z)
        prob_mean.append(d.mean())
        prob_std.append(d.std())

    poly = PolyCollection(verts)

    poly.set_alpha(0.7)
    ax.add_collection3d(poly, zs=valid_times, zdir='y')
    ax.set_xlabel('Log(Event Size Ratio)')
    ax.set_xlim3d(x_min,x_max)
    ax.set_zlabel('P(Event Size Ratio)')
    ax.set_zlim3d(0,1)
    ax.set_ylabel('Timestep in Avalanche')
    ax.set_ylim3d(1,time_steps)

    ax = fig.add_subplot(122)
    ax.errorbar(valid_times, prob_mean, yerr=prob_std)
    ax.set_ylabel('Log(Event Size Ratio)')
    ax.set_ylim(x_min,x_max)
    ax.set_xlabel('Timestep in Avalanche')
    plt.show()
