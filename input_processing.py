def mrc_fieldtrip_import(filename):
    from scipy.io import loadmat
    from numpy import where
    matfile = loadmat(filename)
    data = matfile['D']['trial'][0][0][0][0]
    activity_starts_index = where(data[:306,:].sum(0)!=0)[0][0]
    data = data[:, activity_starts_index:]

    output = {}
    output['magnetometer'] = data[0:102]
    output['gradiometer'] = data[102:306]
    return output

def mrc_raw_import(filename):
    from scipy.io import loadmat
    from numpy import where
    matfile = loadmat(filename)
    data = matfile['D']
    activity_starts_index = where(data[:306,:].sum(0)!=0)[0][0]
    data = data[:, activity_starts_index:]

    output = {}
    output['magnetometer'] = data[0:102]
    output['gradiometer'] = data[102:306]
    return output

def riken_import(directory):
    from scipy.io import loadmat
    from numpy import empty
    from os import listdir
    
    directory_files = listdir(directory)
    n_channels =len([s for s in directory_files if 'ECoG_ch' in s])

    file_base = directory+'/ECoG_ch'
    variable_base = 'ECoGData_ch'

    f = str.format('{0}{1}.mat', file_base, 1)
    v = str.format('{0}{1}', variable_base, 1)
    n_datapoints = loadmat(f)[v].shape[1]

    monkey_data = empty((n_channels,n_datapoints))

    for i in range(n_channels):
        f = str.format('{0}{1}.mat', file_base, i+1)
	v = str.format('{0}{1}', variable_base, i+1)
	monkey_data[i,:] = loadmat(f)[v]
    return monkey_data


def write_to_HDF5(data, file_name, condition, sampling_rate, \
        window='blackmanharris', taps=513,\
        group_name='', species='', location='', number_in_group='',\
        amplitude=False, displacement_aucs=False, amplitude_aucs=False,\
        bands = ('raw', 'delta', 'theta', 'alpha', 'beta', 'gamma', 'high-gamma', 'broad')):
    import h5py
    from neuroscience import neuro_band_filter
    from criticality import area_under_the_curve, fast_amplitude
    from time import gmtime, strftime, clock
    from numpy import concatenate, zeros
    
    filter_type = 'FIR'
    version = 'filter_'+filter_type+'_'+str(taps)+'_'+window
    
    f = h5py.File(file_name+'.hdf5')

    try:
        list(f[condition])
    except KeyError:
        f.create_group(condition)
        pass
    
    for band in bands:
        print 'Processing '+band
        if band=='raw':
            if 'raw' not in list(f[condition]):
                f.create_group(condition+'/raw')

            tic = clock()

            if 'displacement' not in list(f[condition+'/raw']):
                f.create_dataset(condition+'/raw/displacement', data=data)

            if amplitude and 'amplitude' not in list(f[condition+'/raw']):
                data_amplitude = fast_amplitude(data)
                f.create_dataset(condition+'/raw/amplitude', data=data_amplitude)
            if displacement_aucs and 'displacement_aucs' not in list(f[condition+'/raw']):
                data_displacement_aucs = area_under_the_curve(data)
                f.create_dataset(condition+'/raw/displacement_aucs', data=data_displacement_aucs)
            if amplitude_aucs and 'amplitude_aucs' not in list(f[condition+'/raw']):
                data_amplitude_aucs = area_under_the_curve(data_amplitude)
                f.create_dataset(condition+'/raw/amplitude_aucs', data=data_amplitude_aucs)

            toc = clock()
            print toc-tic
            continue

        print 'Filtering, '+str(data.shape[-1])+' time points' 
        filtered_data, frequency_range = neuro_band_filter(data, band, sampling_rate=sampling_rate, taps=taps, window_type=window)

        if version not in list(f[condition]):
            f.create_group(condition+'/'+version)
        
        if 'displacement' not in list(f[condition+'/'+version]):
            f.create_dataset(condition+'/'+version+'/displacement', data=filtered_data)

        if amplitude and 'amplitude' not in list(f[condition+'/'+version]):
            print 'Fast amplitude, '+str(d.shape[-1])+' time points'
            tic = clock()
            data_amplitude = fast_amplitude(filtered_data)
            f.create_dataset(condition+'/'+version+'/amplitude', data=data_amplitude)
            toc = clock()
            print toc-tic
        elif amplitude:
            data_amplitude = f[condition+'/'+version+'/amplitude'][:,:]

        if displacement_aucs and 'displacement_aucs' not in list(f[condition+'/'+version]):
            print 'Area under the curve, displacement'
            tic = clock()
            data_displacement_aucs = area_under_the_curve(filtered_data)
            f.create_dataset(condition+'/'+version+'/displacement_aucs', data=data_displacement_aucs)
            toc = clock()
            print toc-tic

        if amplitude_aucs and 'amplitude_aucs' not in list(f[condition+'/'+version]):
            print 'Fast amplitude, '+str(d.shape[-1])+' time points'
            tic = clock()
            data_amplitude_aucs = area_under_the_curve(data_amplitude)
            f.create_dataset(condition+'/'+version+'/amplitude_aucs', data=data_amplitude_aucs)
            toc = clock()
            print toc-tic
        
        f[condition+'/'+version+'/'+band].attrs['frequency_range'] = frequency_range 
        f[condition+'/'+version+'/'+band].attrs['processing_date'] = strftime("%Y-%m-%d", gmtime())
    
    f[condition+'/'+version].attrs['filter_type'] = filter_type
    f[condition+'/'+version].attrs['window'] = window
    f[condition+'/'+version].attrs['taps'] = taps
    f.attrs['group_name']=group_name
    f.attrs['number_in_group']=number_in_group
    f.attrs['species'] = species
    f.attrs['location']=location
    
    f.close()
    return
