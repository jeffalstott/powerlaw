import os
import HDF5_conversion
import human_import

path = '/home/jja34/public_html/MEG_Data/Rest/Group1'
dirList=os.listdir(path)

for fname in dirList:
    if fname[0]=='f':
        filename = path+'/'+fname
        data = human_import.fieldtrip_import(filename)
        subject_id = fname[31:34]
        eyes = fname[26:30]
        visit = fname[37]
        print fname
        print subject_id+eyes+visit
        output_file = '/work/imaging8/jja34/MEG_Study/MEG_Data/Group1/Subject'+subject_id
        condition = 'rest/'+eyes+'/magnetometer'
        HDF5_conversion.convert(data['magnetometer'],output_file, condition, 250.0)
        condition = 'rest/'+eyes+'/gradiometer'
        HDF5_conversion.convert(data['gradiometer'],output_file, condition, 250.0)


