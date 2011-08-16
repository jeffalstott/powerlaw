import h5py
from criticality import neuro_band_filter
from scipy.signal import hilbert
from time import gmtime, strftime

subject_name = 'Monkey_K2'
version = 'rest/filter_version0'
sampling_rate=1000.0


filter_type = 'FIR'
window = 'blackmanharris'
taps = 511.0

f = h5py.File(subject_name+'.hdf5')
f.create_group(version)
f[version].attrs['filter_type'] = filter_type
f[version].attrs['window'] = window
f[version].attrs['taps'] = taps

print 'Processing raw data'
hd = abs(hilbert(data))
f.create_dataset(version+'/raw/displacement', data=data)
f.create_dataset(version+'/raw/amplitude', data=hd)

bands = ('delta', 'theta', 'alpha', 'beta', 'gamma', 'high-gamma', 'broad')

for band in bands:
    print band
    d, frequency_range = neuro_band_filter(data, band, sampling_rate=sampling_rate, taps=taps, window_type=window)
    f.create_dataset(version+'/'+band+'/displacement', data=d)
    hd = abs(hilbert(d))
    f.create_dataset(version+'/'+band+'/amplitude', data=hd)
    f[version+'/'+band].attrs['frequency_range'] = frequency_range 
    f[version+'/'+band].attrs['processing_date'] = strftime("%Y-%m-%d", gmtime())

f.close()
