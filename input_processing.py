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
    if 'ECoG_and_Event.mat' in directory_files:
        mat = loadmat(directory+'ECoG_and_Event.mat')
        monkey_data = mat['ECoG'].astype(float)
        return monkey_data

    elif 'ECoG_ch1.mat' in directory_files:
        n_channels =len([s for s in directory_files if 'ECoG_ch' in s])

        file_base = directory+'ECoG_ch'
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
    else:
        print("Unsupported data format")
        return


def write_to_HDF5(data, file_name, condition, sampling_rate, \
        window='blackmanharris', taps=513, filter_type='FIR',\
        group_name='', species='', location='', number_in_group='', name='', date='',\
        amplitude=False, displacement_aucs=False, amplitude_aucs=False,\
        overwrite=False,\
        bands = ('raw', 'delta', 'theta', 'alpha', 'beta', 'gamma', 'high-gamma', 'broad'),
        downsample='nyquist'):
    import h5py
    from neuroscience import neuro_band_filter
    from avalanches import area_under_the_curve, fast_amplitude
    from time import gmtime, strftime, clock

    if downsample==False:
        downsample=sampling_rate
    
    version = 'filter_'+filter_type+'_'+str(taps)+'_'+window+'_ds-'+str(downsample)
    
    f = h5py.File(file_name+'.hdf5')

    try:
        versions = list(f[condition])
        if version not in versions:
            f.create_group(condition+'/'+version)
    except KeyError:
        f.create_group(condition+'/'+version)
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

        if band not in list(f[condition+'/'+version]):
            f.create_group(condition+'/'+version+'/'+band)
        
        if 'displacement' not in list(f[condition+'/'+version+'/'+band]):
            print 'Filtering, '+str(data.shape[-1])+' time points' 
            filtered_data, frequency_range, downsampled_rate = neuro_band_filter(data, band, sampling_rate=sampling_rate, taps=taps, window_type=window, downsample=downsample)
            f.create_dataset(condition+'/'+version+'/'+band+'/displacement', data=filtered_data)
        elif overwrite:
            print 'Filtering, '+str(data.shape[-1])+' time points' 
            filtered_data, frequency_range, downsampled_rate = neuro_band_filter(data, band, sampling_rate=sampling_rate, taps=taps, window_type=window, downsample=downsample)
            f[condition+'/'+version+'/'+band+'/displacement']=filtered_data
        elif amplitude_aucs or amplitude or displacement_aucs:
            filtered_data = f[condition+'/'+version+'/'+band+'/displacement'][:,:]
        else:
            continue

        if amplitude and 'amplitude' not in list(f[condition+'/'+version+'/'+band]):
            print 'Fast amplitude, '+str(filtered_data.shape[-1])+' time points'
            tic = clock()
            data_amplitude = fast_amplitude(filtered_data)
            f.create_dataset(condition+'/'+version+'/'+band+'/amplitude', data=data_amplitude)
            toc = clock()
            print toc-tic
        elif amplitude:
            data_amplitude = f[condition+'/'+version+'/'+band+'/amplitude'][:,:]

        if displacement_aucs and 'displacement_aucs' not in list(f[condition+'/'+version+'/'+band]):
            print 'Area under the curve, displacement'
            tic = clock()
            data_displacement_aucs = area_under_the_curve(filtered_data)
            f.create_dataset(condition+'/'+version+'/'+band+'/displacement_aucs', data=data_displacement_aucs)
            toc = clock()
            print toc-tic

        if amplitude_aucs and 'amplitude_aucs' not in list(f[condition+'/'+version+'/'+band]):
            print 'Area under the curve, amplitude'
            tic = clock()
            data_amplitude_aucs = area_under_the_curve(data_amplitude)
            f.create_dataset(condition+'/'+version+'/'+band+'/amplitude_aucs', data=data_amplitude_aucs)
            toc = clock()
            print toc-tic
        
        f[condition+'/'+version+'/'+band].attrs['frequency_range'] = frequency_range
        f[condition+'/'+version+'/'+band].attrs['downsampled_rate'] = downsampled_rate
        f[condition+'/'+version+'/'+band].attrs['processing_date'] = strftime("%Y-%m-%d", gmtime())
    
    f[condition+'/'+version].attrs['filter_type'] = filter_type
    f[condition+'/'+version].attrs['window'] = window
    f[condition+'/'+version].attrs['taps'] = taps
    f.attrs['group_name']=group_name
    f.attrs['number_in_group']=number_in_group
    f.attrs['species'] = species
    f.attrs['location']=location
    f.attrs['name']=name
    f[condition].attrs['date']=date
    
    f.close()
    return

def HDF5_filter(file,\
        window='hamming', taps=25, filter_type='FIR',\
        amplitude=False, displacement_aucs=False, amplitude_aucs=False,\
        overwrite=False,\
        bands = ('raw', 'delta', 'theta', 'alpha', 'beta', 'gamma', 'high-gamma', 'broad'),
        downsample='nyquist'):

    from neuroscience import neuro_band_filter
    from avalanches import area_under_the_curve, fast_amplitude
    from time import gmtime, strftime, clock
    import h5py

    if type(file)!=h5py._hl.group.Group:
        return

    for i in file.keys():
        if i.startswith('filter'):
            continue
        elif not i.startswith('raw'):
            HDF5_filter(file[i])
        else:
            if 'displacement' not in file[i].keys():
                return
            else:
#At this point we know there is a 'raw' directory with a 'displacement' in it,
# so we can filter!
                if downsample==False:
                    downsample=sampling_rate

                version = 'filter_'+filter_type+'_'+str(taps)+'_'+window+'_ds-'+str(downsample)
                if version not in file.keys():
                    file.create_group(version)
                file[version].attrs['filter_type'] = filter_type
                file[version].attrs['window'] = window
                file[version].attrs['taps'] = taps

                data = file['raw/displacement'][:,:]

                for band in bands:
                    print 'Processing '+band
                    if band=='raw':
                        if amplitude and 'amplitude' not in file['raw'].keys():
                            data_amplitude = fast_amplitude(data)
                            file.create_dataset('/raw/amplitude', data=data_amplitude)
                        if displacement_aucs and 'displacement_aucs' not in file['raw'].keys():
                            data_displacement_aucs = area_under_the_curve(data)
                            file.create_dataset('/raw/displacement_aucs', data=data_displacement_aucs)
                        if amplitude_aucs and 'amplitude_aucs' not in file['raw'].keys():
                            data_amplitude_aucs = area_under_the_curve(data_amplitude)
                            file.create_dataset('/raw/amplitude_aucs', data=data_amplitude_aucs)
                        continue

                    if band not in file[version].keys():
                        file.create_group(version+'/'+band)

                    if 'displacement' not in file[version+'/'+band].keys():
                        print 'Filtering, '+str(data.shape[-1])+' time points' 
                        filtered_data, frequency_range, downsampled_rate = neuro_band_filter(data, band, sampling_rate=sampling_rate, taps=taps, window_type=window, downsample=downsample)
                        file.create_dataset(version+'/'+band+'/displacement', data=filtered_data)
                    elif overwrite:
                        print 'Filtering, '+str(data.shape[-1])+' time points' 
                        filtered_data, frequency_range, downsampled_rate = neuro_band_filter(data, band, sampling_rate=sampling_rate, taps=taps, window_type=window, downsample=downsample)
                        file.create_dataset(version+'/'+band+'/displacement', data=filtered_data)
                    elif amplitude_aucs or amplitude or displacement_aucs:
                        filtered_data = file[version+'/'+band+'/displacement'][:,:]
                    else:
                        continue

                    if amplitude and 'amplitude' not in file[version+'/'+band].keys():
                        print 'Fast amplitude, '+str(filtered_data.shape[-1])+' time points'
                        tic = clock()
                        data_amplitude = fast_amplitude(filtered_data)
                        file.create_dataset(version+'/'+band+'/amplitude', data=data_amplitude)
                        toc = clock()
                        print toc-tic
                    elif amplitude_aucs:
                        data_amplitude = file[version+'/'+band+'/amplitude'][:,:]

                    if displacement_aucs and 'displacement_aucs' not in file[version+'/'+band].keys():
                        print 'Area under the curve, displacement'
                        tic = clock()
                        data_displacement_aucs = area_under_the_curve(filtered_data)
                        file.create_dataset(version+'/'+band+'/displacement_aucs', data=data_displacement_aucs)
                        toc = clock()
                        print toc-tic

                    if amplitude_aucs and 'amplitude_aucs' not in file[version+'/'+band].keys():
                        print 'Area under the curve, amplitude'
                        tic = clock()
                        data_amplitude_aucs = area_under_the_curve(data_amplitude)
                        file.create_dataset(version+'/'+band+'/amplitude_aucs', data=data_amplitude_aucs)
                        toc = clock()
                        print toc-tic

                    file[version+'/'+band].attrs['frequency_range'] = frequency_range
                    file[version+'/'+band].attrs['downsampled_rate'] = downsampled_rate
                    file[version+'/'+band].attrs['processing_date'] = strftime("%Y-%m-%d", gmtime())
                return
