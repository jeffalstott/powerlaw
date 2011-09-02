import os
from input_processing import mrc_fieldtrip_import, write_to_HDF5

path = '/home/jja34/public_html/MEG_Data/Rest/Group1'
dirList=os.listdir(path)

bands = ('delta', 'theta', 'alpha', 'beta', 'gamma', 'high-gamma', 'raw', 'broad')

for fname in dirList:
    components = fname.split('_')
    if components[0] == 'fieldtrip' and (components[6]=='001' or components[6]=='002'):
        filename = path+'/'+fname
        data = mrc_fieldtrip_import(filename)
        subject_id = components[6]
        eyes = components[5]
        visit = components[8]
        print fname
        print subject_id+eyes+visit
        output_file = '/work/imagingA/jja34/MEG_Study/Data/Group1/Subject'+subject_id
        condition = visit+'/rest/'+eyes+'/magnetometer'
        write_to_HDF5(data['magnetometer'],output_file, condition, 250.0, bands=bands)
        condition = visit+'/rest/'+eyes+'/gradiometer'
        write_to_HDF5(data['gradiometer'],output_file, condition, 250.0, bands=bands)


