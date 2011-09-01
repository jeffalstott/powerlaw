import criticality
import h5py
from numpy import arange
import os

bins = arange(10)+1
percentiles = [.9, .95, .99, .999]
event_methods = ['amplitude', 'displacement']
cascade_methods = ['grid', 'gap']
subsamples = [('all', 'all')]


path = '/work/imaging8/jja34/MEG_Study/MEG_Data/Group1'
conditions = ['2/rest/open/gradiometer', '2/rest/open/magnetometer',\
        '2/rest/shut/gradiometer', '2/rest/shut/magnetometer',\
        '3/rest/open/gradiometer', '3/rest/open/magnetometer',\
        '3/rest/shut/gradiometer', '3/rest/shut/magnetometer']
filter = 'filter_FIR_513_blackmanharris'
dirList=os.listdir(path)


for fname in dirList:
    subject = fname
    file = path+'/'+subject
    print file
    f = h5py.File(file)
    for condition in conditions:
        base = condition+'/'+filter
        print base
        for band in list(f[base]):
            data = f[base+'/'+band]
            criticality.avalanche_analyses(data, bins, percentiles, event_methods, cascade_methods, subsamples)

