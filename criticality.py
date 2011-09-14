from numpy import array, where, log2
from bisect import bisect_left
from sys import float_info

def avalanche_analysis(data, data_amplitude=False, data_displacement_aucs=False, data_amplitude_aucs=False, \
        bin_width=1, percentile=.99, \
        event_method='amplitude', cascade_method='grid', \
        spatial_sample='all', spatial_sample_name=False,\
        temporal_sample='all', temporal_sample_name=False,\
        write_to_HDF5=False, overwrite=False):
    """docstring for avalanche_analysis  """

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
        version = 'b-'+str(bin_width)+'_p-'+str(percentile)[2:]+'_e-'+event_method + '_c-'+ cascade_method +'_s-'+spatial_sample_name+'_t-'+temporal_sample_name
        if not overwrite and 'avalanches' in list(data) and version in list(data['avalanches']):
            return 'Calculation aborted for version '+version+' as results already exist and the option overwrite=False'

    metrics = {}
    metrics['bin_width'] = bin_width
    metrics['percentile'] = percentile
    metrics['event_method'] = event_method
    metrics['cascade_method'] = cascade_method
    metrics['spatial_sample'] = spatial_sample_name 
    metrics['temporal_sample'] = temporal_sample_name 

    m = find_events(data, data_amplitude, data_displacement_aucs,  data_amplitude_aucs, percentile, event_method, spatial_sample, temporal_sample)
    metrics.update(m)

    starts, stops = find_cascades(metrics['event_times'], bin_width, cascade_method)

    metrics['starts'] = starts
    metrics['stops'] = stops
    metrics['durations'] = (stops-starts).astype(float)
    metrics['durations_silences'] = (starts[1:]-stops[:-1]).astype(float)

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
        attributes = ('bin_width', 'percentile', 'event_method', 'cascade_method',\
                'spatial_sample', 'temporal_sample')
        for k in attributes:
            results_subgroup.attrs[k] = metrics[k]
        for k in metrics:
            if k not in attributes:
                if len(metrics[k])!=0:
                    results_subgroup.create_dataset(k, data=metrics[k])
                else:
                    results_subgroup.create_dataset(k, data=array([0]))
        return


def find_events(data, data_amplitude=False, data_displacement_aucs=False, data_amplitude_aucs=False,\
        percentile=.99, event_method='amplitude', spatial_sample='all', temporal_sample='all'):
    """find_events does things"""
    from scipy.signal import hilbert
    from scipy.stats import scoreatpercentile
    from numpy import ndarray, transpose, diff
    import h5py

    #See if we received a reference to section of an HDF5 file. If so, pull what data is available
    if type(data)==h5py._hl.group.Group:
        data_displacement = data['displacement'][:,:]
        if 'amplitude' in data:
            data_amplitude = data['amplitude'][:,:]
        if 'displacement_aucs' in data:
            data_displacement_aucs = data['displacement_aucs'][:,:]
        if 'amplitude_aucs' in data:
            data_amplitude_aucs = data['amplitude_aucs'][:,:]
    else:
        data_displacement = data
    
    n_rows, n_columns = data_displacement.shape
    data_displacement = data_displacement-data_displacement.mean(1).reshape(n_rows,1)

    #If we don't have amplitude or area_under_the_curve information yet, calculate it
    if type(data_amplitude)!=ndarray:
        data_amplitude = abs(hilbert(data_displacement))
    if type(data_displacement_aucs)!=ndarray:
        data_displacement_aucs = area_under_the_curve(data_displacement)
    if type(data_amplitude_aucs)!=ndarray:
        data_amplitude_aucs = area_under_the_curve(data_amplitude)

    #If we're not using all sensors, take the spatial sample Now
    if spatial_sample!='all':
        data_displacement = data_displacement[spatial_sample,:]
        data_amplitude = data_amplitude[spatial_sample,:]
        data_displacement_aucs = data_displacement_aucs[spatial_sample,:]
        data_amplitude_aucs = data_amplitude_aucs[spatial_sample,:]

    if event_method == 'amplitude':
        signal = data_amplitude
    elif event_method == 'displacement':
        signal = abs(data_displacement)
    elif event_method == 'displacement_up':
        signal = data_displacement
    elif event_method == 'displacement_down':
        signal = data_displacement*-1.0
    else:
        print 'Please select a supported event detection method (amplitude or displacement)'

    #scoreatpercentile only computes along the first dimension, so we transpose the 
    #(channels, times) matrix to a (times, channels) matrix. This is also useful for
    #applying the threshold, which is #channels long. We just need to make sure to 
    #invert back the coordinate system when we assign the results, which we do 
    threshold = scoreatpercentile(transpose(signal), percentile*100)
    times, channels = where(transpose(signal)>threshold)

    #If we're not using all time points, take the temporal sample now
    if temporal_sample!='all':
        allowed_times = []
        for i in len(times):
            q = bisect_left(temporal_sample, times[i])
            if q!=len(temporal_sample) and temporal_sample[q]==times[i]:
                allowed_times.append(i)
        times = times[allowed_times]
        channels = channels[allowed_times]

    displacements = data_displacement[channels, times]
    amplitudes = data_amplitude[channels,times]
    interevent_intervals = diff(times)

    event_amplitude_aucs = data_amplitude_aucs[channels, times]
    event_displacement_aucs = data_displacement_aucs[channels, times]

    output_metrics = { \
            'event_times': times, \
            'event_channels': channels,\
            'event_displacements': displacements,\
            'event_amplitudes': amplitudes,\
            'event_amplitude_aucs': event_amplitude_aucs,\
            'event_displacement_aucs': event_displacement_aucs,\
            'interevent_intervals': interevent_intervals,\
            }
    return output_metrics

def find_cascades(event_times, bin_width=1, method='grid'):
    """find_events does things"""
    from numpy import diff, concatenate

    if method=='gap':
        starts = array([event_times[0]])
        stops = array([event_times[-1]])
        changes = where(diff(event_times)>=bin_width+1)[0]
        starts = concatenate((starts, event_times[changes+1]))
        stops = concatenate((event_times[changes], stops))

    elif method=='grid':
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
            (2*input_metrics['bin_width']):
                sigma_amplitudes = sigma_events = \
                        sigma_displacements = sigma_amplitude_aucs = \
                        0
    else:
        first_bin = bisect_left( \
                input_metrics['event_times'], \
                (input_metrics['starts'][avalanche_number] \
                +input_metrics['bin_width'])\
                )-1
        second_bin = bisect_left( \
                input_metrics['event_times'], \
                (input_metrics['starts'][avalanche_number] \
                +2*input_metrics['bin_width'])\
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

def avalanche_statistics(metrics, write_to_database=False, overwrite=False, analysis_id=None,\
        filter_id=None, subject_id=None, task_id=None, experiment_id=None, sensor_id=None, recording_id=None):
    from scipy.stats import mode, linregress
    from numpy import empty, unique, median
    from plfit import plfit
    
    if write_to_database:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        import database_classes as dc
        engine = create_engine(write_to_database, echo=False)
        Session = sessionmaker(bind=engine)
        session = Session()
	
        avalanche_analysis = session.query(dc.Avalanche).filter_by(\
                id=analysis_id).first()

	#If there are statistics already calculated, and we're not overwriting, then end.
        if avalanche_analysis.fits and not overwrite:
            return

    statistics = {}
    times_within_avalanche = unique(metrics['event_times_within_avalanche'])
    j = empty(times_within_avalanche.shape)

    for k in metrics:
        if k.startswith('sigma'):
            statistics[k]=metrics[k].mean()

            if write_to_database:
                setattr(avalanche_analysis, k, statistics[k])

        if k.startswith('interevent_intervals'):
            statistics[k+'_mean']=metrics[k].mean()
            statistics[k+'_median']=median(metrics[k])
            statistics[k+'_mode']=mode(metrics[k])[0][0]

            if write_to_database:
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

            if write_to_database:
                setattr(avalanche_analysis, k+'_slope', statistics[k]['slope'])
                setattr(avalanche_analysis, k+'_R', statistics[k]['R'])
                setattr(avalanche_analysis, k+'_p', statistics[k]['p'])

        elif k.startswith('duration') or k.startswith('size'):
            statistics[k]={}
            statistics[k]['power_law']={}
            fit=plfit(metrics[k], usefortran=False, quiet=True, silent=True)
            statistics[k]['power_law']['parameter1_name']='alpha'
            statistics[k]['power_law']['parameter1_value']=fit._alpha
            statistics[k]['power_law']['parameter2_name']='error'
            statistics[k]['power_law']['parameter2_value']=fit._alphaerr
            statistics[k]['power_law']['parameter3_name']=None
            statistics[k]['power_law']['parameter3_value']=None
            statistics[k]['power_law']['xmin']=fit._xmin
            statistics[k]['power_law']['loglikelihood']=fit._alphaerr
            statistics[k]['power_law']['KS']=fit._ks
            statistics[k]['power_law']['p']=fit._ks_prob

            statistics[k]['lognormal']={}
            fit.lognormal(doprint=False)
            statistics[k]['lognormal']['parameter1_name']='shape'
            statistics[k]['lognormal']['parameter1_value']=fit.lognormal_dist.args[0]
            statistics[k]['lognormal']['parameter2_name']='location'
            statistics[k]['lognormal']['parameter2_value']=fit.lognormal_dist.args[1]
            statistics[k]['lognormal']['parameter3_name']='scale'
            statistics[k]['lognormal']['parameter3_value']=fit.lognormal_dist.args[2]
            statistics[k]['lognormal']['xmin']=None
            statistics[k]['lognormal']['loglikelihood']=-1*fit.lognormal_likelihood
            statistics[k]['lognormal']['KS']=fit.lognormal_ksD
            statistics[k]['lognormal']['p']=fit.lognormal_ksP

            if write_to_database:
                fit_variables = ('parameter1_name', 'parameter1_value', \
                        'parameter2_name', 'parameter2_value', \
                        'parameter3_name', 'parameter3_value', \
                        'xmin', 'loglikelihood', 'KS', 'p')

                power_law_fit = dc.Fit(analysis_type='avalanches',\
                        variable=k, distribution='power_law',\
                        subject_id=subject_id, task_id=task_id, experiment_id=experiment_id,\
                        sensor_id=sensor_id, recording_id=recording_id, filter_id=filter_id,\
                        analysis_id=analysis_id)

                lognormal_fit = dc.Fit(analysis_type='avalanches',\
                        variable=k, distribution='lognormal',\
                        subject_id=subject_id, task_id=task_id, experiment_id=experiment_id,\
                        sensor_id=sensor_id, recording_id=recording_id, filter_id=filter_id,\
                        analysis_id=analysis_id)

                for variable in fit_variables:
                    if statistics[k]['power_law'][variable]==float('inf'):
                        setattr(power_law_fit,variable, 1*10**float_info.max_10_exp)
                    elif statistics[k]['power_law'][variable]==-float('inf'):
                        setattr(power_law_fit,variable, -1*10**float_info.max_10_exp)
                    else:
                        setattr(power_law_fit,variable, statistics[k]['power_law'][variable])

                    if statistics[k]['lognormal'][variable]==float('inf'):
                        setattr(lognormal_fit,variable, 1*10**float_info.max_10_exp)
                    elif statistics[k]['lognormal'][variable]==-float('inf'):
                        setattr(lognormal_fit,variable, -1*10**float_info.max_10_exp)
                    else:
                        setattr(lognormal_fit,variable, statistics[k]['lognormal'][variable])
                avalanche_analysis.fits.append(power_law_fit)
                avalanche_analysis.fits.append(lognormal_fit)

    if write_to_database:
        session.add(avalanche_analysis)
        session.commit()
    return statistics

def avalanche_analyses(data,\
        bins, percentiles, event_methods, cascade_methods, \
        spatial_samples, temporal_samples, \
        spatial_sample_names=False, temporal_sample_names=False, \
        write_to_HDF5=False, overwrite_HDF5=False,\
        write_to_database=False, overwrite_database=False,\
        filter_id=None, subject_id=None, task_id=None, experiment_id=None, sensor_id=None, recording_id=None,\
        data_amplitude=False, data_displacement_aucs=False, data_amplitude_aucs=False,\
        verbose=False):

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
    results = {}

    if write_to_database:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        import database_classes as dc
        engine = create_engine(write_to_database, echo=False)
        Session = sessionmaker(bind=engine)
        session = Session()

    parameter_space = [(b,p,e,c,s,sn,t,tn) for b in bins for p in percentiles \
            for e in event_methods for c in cascade_methods \
            for s,sn in spatial_samples \
            for t,tn in temporal_samples]

    for b,p,e,c,s,sn,t,tn in parameter_space:
        parameters = str(b)+'_'+str(p)+'_'+str(e)+'_'+str(c)+'_'+str(sn)+'_'+str(tn)
        results[parameters] = {}

        if verbose:
            print parameters

        if write_to_database:
            analysis = session.query(dc.Avalanche).filter_by(\
                    filter_id=filter_id, spatial_sample=sn, temporal_sample=tn,\
                    threshold_mode='percentile', threshold_level=p, \
                    time_scale=b, event_method=e, cascade_method=c).first()
            print analysis
#            import pdb; pdb.set_ttrace()
            #If we're not overwriting the database, and there is a previous analysis with saved statistics, then go on to the next set of parameters
            if not overwrite_database and analysis and analysis.fits:
                continue

            if not analysis:
                analysis = dc.Avalanche(\
                    filter_id=filter_id, spatial_sample=sn, temporal_sample=tn,\
                    threshold_mode='percentile', threshold_level=p, \
                    time_scale=b, event_method=e, cascade_method=c,\
                    subject_id=subject_id, task_id=task_id, experiment_id=experiment_id,\
                    sensor_id=sensor_id, recording_id=recording_id,\
                    fits = [])
                session.add(analysis)
                session.commit()

            analysis_id=analysis.id

        metrics = avalanche_analysis(data,\
            data_amplitude=data_amplitude, \
            data_displacement_aucs=data_displacement_aucs,\
            data_amplitude_aucs=data_amplitude_aucs,\
            bin_width=b, percentile=p,\
            event_method=e, cascade_method=c,\
            spatial_sample=s, spatial_sample_name=sn,\
            temporal_sample=t, temporal_sample_name=tn,\
            write_to_HDF5=write_to_HDF5, overwrite=overwrite_HDF5)

        statistics = avalanche_statistics(metrics, \
                write_to_database=write_to_database, analysis_id=analysis_id,\
                subject_id=subject_id, task_id=task_id, experiment_id=experiment_id,\
                sensor_id=sensor_id, recording_id=recording_id, filter_id=filter_id)

        results[parameters]['metrics'] = metrics 
        results[parameters]['statistics'] = statistics

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






