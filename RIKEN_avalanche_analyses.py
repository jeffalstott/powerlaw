import criticality
import h5py
from numpy import arange

bins = arange(10)+1
percentiles = [.9, .95, .99, .999]
event_methods = ['amplitude', 'displacement']
cascade_methods = ['grid', 'gap']
subsamples = [('all', 'all')]

monkey = '/scratch/alstottj/Monkey_A.hdf5'
print monkey
f = h5py.File(monkey)
base = 'food_tracking0/filter_FIR_513_blackmanharris'
print base
for band in list(f[base]):
    data = f[base+'/'+band]
    criticality.avalanche_analyses(data, bins, percentiles, event_methods, cascade_methods, subsamples)

base = 'food_tracking1/filter_FIR_513_blackmanharris'
print base
for band in list(f[base]):
    data = f[base+'/'+band]
    criticality.avalanche_analyses(data, bins, percentiles, event_methods, cascade_methods, subsamples)

base = 'food_tracking2/filter_FIR_513_blackmanharris'
print base
for band in list(f[base]):
    data = f[base+'/'+band]
    criticality.avalanche_analyses(data, bins, percentiles, event_methods, cascade_methods, subsamples)

base = 'food_tracking3/filter_FIR_513_blackmanharris'
print base
for band in list(f[base]):
    data = f[base+'/'+band]
    criticality.avalanche_analyses(data, bins, percentiles, event_methods, cascade_methods, subsamples)

base = 'food_tracking4/filter_FIR_513_blackmanharris'
print base
for band in list(f[base]):
    data = f[base+'/'+band]
    criticality.avalanche_analyses(data, bins, percentiles, event_methods, cascade_methods, subsamples)
f.close()

monkey = '/scratch/alstottj/Monkey_K1.hdf5'
print monkey
f = h5py.File(monkey)
base = 'food_tracking0/filter_FIR_513_blackmanharris'
print base
for band in list(f[base]):
    data = f[base+'/'+band]
    criticality.avalanche_analyses(data, bins, percentiles, event_methods, cascade_methods, subsamples)

base = 'food_tracking1/filter_FIR_513_blackmanharris'
print base
for band in list(f[base]):
    data = f[base+'/'+band]
    criticality.avalanche_analyses(data, bins, percentiles, event_methods, cascade_methods, subsamples)

base = 'food_tracking2/filter_FIR_513_blackmanharris'
print base
for band in list(f[base]):
    data = f[base+'/'+band]
    criticality.avalanche_analyses(data, bins, percentiles, event_methods, cascade_methods, subsamples)
f.close()

monkey = '/scratch/alstottj/Monkey_K2.hdf5'
print monkey
f = h5py.File(monkey)
base = 'emotional_movie/filter_FIR_513_blackmanharris'
print base
for band in list(f[base]):
    data = f[base+'/'+band]
    criticality.avalanche_analyses(data, bins, percentiles, event_methods, cascade_methods, subsamples)

base = 'rest/filter_FIR_513_blackmanharris'
print base
for band in list(f[base]):
    data = f[base+'/'+band]
    criticality.avalanche_analyses(data, bins, percentiles, event_methods, cascade_methods, subsamples)

base = 'social_competition/filter_FIR_513_blackmanharris'
print base
for band in list(f[base]):
    data = f[base+'/'+band]
    criticality.avalanche_analyses(data, bins, percentiles, event_methods, cascade_methods, subsamples)

base = 'visual_grating/filter_FIR_513_blackmanharris'
print base
for band in list(f[base]):
    data = f[base+'/'+band]
    criticality.avalanche_analyses(data, bins, percentiles, event_methods, cascade_methods, subsamples)

base = 'anesthesia/filter_FIR_513_blackmanharris'
print base
for band in list(f[base]):
    data = f[base+'/'+band]
    criticality.avalanche_analyses(data, bins, percentiles, event_methods, cascade_methods, subsamples)
