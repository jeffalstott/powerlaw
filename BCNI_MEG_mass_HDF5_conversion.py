import os
from input_processing import mrc_raw_import, write_to_HDF5

path = '/home/jja34/public_html/MEG_Data/Rest/Group2'
dirList=os.listdir(path)

bands = ('delta', 'theta', 'alpha', 'beta', 'gamma', 'high-gamma', 'raw', 'broad')

for fname in dirList:
    if fname[0]=='r':
        filename = path+'/'+fname
        data = mrc_raw_import(filename)
        subject_id = fname[31:34]
        eyes = fname[26:30]
        visit = fname[37]
        print fname
        print subject_id+eyes+visit
        output_file = '/work/imaging8/jja34/MEG_Study/MEG_Data/Group2/Subject'+subject_id
        condition = visit+'/rest/'+eyes+'/magnetometer'
        write_to_HDF5(data['magnetometer'],output_file, condition, 250.0, bands=bands)
        condition = visit+'/rest/'+eyes+'/gradiometer'
        write_to_HDF5(data['gradiometer'],output_file, condition, 250.0, bands=bands)


