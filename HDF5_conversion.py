def convert(data, file_name, condition, sampling_rate, bands = ('delta', 'theta', 'alpha', 'beta', 'gamma', 'high-gamma', 'broad')):
    import h5py
    from neuroscience import neuro_band_filter
    from criticality import area_under_the_curve
    from scipy.signal import hilbert
    from time import gmtime, strftime, clock
    
    #subject_name = 'Monkey_K2'
    #condition = 'anesthesia'
    version = 'filter_version0'
    #sampling_rate=1000.0
    
    
    filter_type = 'FIR'
    window = 'blackmanharris'
    taps = 511.0
    
    f = h5py.File(file_name+'.hdf5')
    f.create_group(condition+'/'+version)
    f[condition+'/'+version].attrs['filter_type'] = filter_type
    f[condition+'/'+version].attrs['window'] = window
    f[condition+'/'+version].attrs['taps'] = taps
    
    print 'Processing raw data'
    data_amplitude = abs(hilbert(data))
    data_displacement_aucs = area_under_the_curve(data)
    data_amplitude_aucs = area_under_the_curve(data_amplitude)
    f.create_dataset(condition+'/raw/displacement', data=data)
    f.create_dataset(condition+'/raw/amplitude', data=data_amplitude)
    f.create_dataset(condition+'/raw/amplitude_aucs', data=data_amplitude_aucs)
    f.create_dataset(condition+'/raw/displacement_aucs', data=data_displacement_aucs)
    
    
    for band in bands:
        print 'Processing '+band
        print 'Filtering' 
        tic = clock()
        d, frequency_range = neuro_band_filter(data, band, sampling_rate=sampling_rate, taps=taps, window_type=window)
        f.create_dataset(condition+'/'+version+'/'+band+'/displacement', data=d)
        toc = clock()
        print toc-tic
        print 'Hilbert Transform'
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
    
    
    f.close()
    return
