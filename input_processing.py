def bcni_fieldtrip_import(filename):
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


def write_to_HDF5(data, file_name, condition, sampling_rate, bands = ('raw', 'delta', 'theta', 'alpha', 'beta', 'gamma', 'high-gamma', 'broad')):
    import h5py
    from neuroscience import neuro_band_filter
    from criticality import area_under_the_curve
    from scipy.signal import hilbert
    from time import gmtime, strftime, clock
    
    #subject_name = 'Monkey_K2'
    #condition = 'anesthesia'
    version = 'filter_FIR_513_blackmanharris'
    #sampling_rate=1000.0
    
    
    filter_type = 'FIR'
    window = 'blackmanharris'
    taps = 513.0
    
    f = h5py.File(file_name+'.hdf5')
    
    
    for band in bands:
        print 'Processing '+band
        if band=='raw':
            tic = clock()
            data_amplitude = abs(hilbert(data))
            data_displacement_aucs = area_under_the_curve(data)
            data_amplitude_aucs = area_under_the_curve(data_amplitude)
            f.create_dataset(condition+'/raw/displacement', data=data)
            f.create_dataset(condition+'/raw/amplitude', data=data_amplitude)
            f.create_dataset(condition+'/raw/amplitude_aucs', data=data_amplitude_aucs)
            f.create_dataset(condition+'/raw/displacement_aucs', data=data_displacement_aucs)
            toc = clock()
            print toc-tic
            continue
        print 'Filtering '+str(data.shape[-1])+' time points' 
        tic = clock()
        d, frequency_range = neuro_band_filter(data, band, sampling_rate=sampling_rate, taps=taps, window_type=window)
        #Make sure length of data isn't prime before handing to hilbert, which can run in O(N^2) time for prime-length datasets
        if d.shape[-1]%2:
            d = d[:,:-1]
        f.create_dataset(condition+'/'+version+'/'+band+'/displacement', data=d)
        toc = clock()
        print toc-tic
        print 'Hilbert Transform '+str(d.shape[-1])+' time points'
        tic = clock()
        if band in ('broad', 'gamma', 'high-gamma'):
            import pdb; pdb.set_trace()
        hd = abs(hilbert(d))
        f.create_dataset(condition+'/'+version+'/'+band+'/amplitude', data=hd)
        toc = clock()
        print toc-tic
        print 'Area under the curve, displacement'
        tic = clock()
        data_displacement_aucs = area_under_the_curve(d)
        f.create_dataset(condition+'/'+version+'/'+band+'/displacement_aucs', data=data_displacement_aucs)
        toc = clock()
        print toc-tic
        print 'Area under the curve, amplitude'
        tic = clock()
        data_amplitude_aucs = area_under_the_curve(hd)
        f.create_dataset(condition+'/'+version+'/'+band+'/amplitude_aucs', data=data_amplitude_aucs)
        toc = clock()
        print toc-tic
        
        f[condition+'/'+version+'/'+band].attrs['frequency_range'] = frequency_range 
        f[condition+'/'+version+'/'+band].attrs['processing_date'] = strftime("%Y-%m-%d", gmtime())
    
    f[condition+'/'+version].attrs['filter_type'] = filter_type
    f[condition+'/'+version].attrs['window'] = window
    f[condition+'/'+version].attrs['taps'] = taps
    
    f.close()
    return
