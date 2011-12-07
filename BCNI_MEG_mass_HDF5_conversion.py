import os
from input_processing import mrc_raw_import, write_to_HDF5

path = '/scratch/alstottj/MRC/For_Filtering/'
output_directory = '/scratch/alstottj/MRC/For_Analysis/'
dirList=os.listdir(path)

bands = ('delta', 'theta', 'alpha', 'beta', 'gamma', 'high-gamma', 'broad')#, 'raw')
window='hamming'
taps=25

for fname in dirList:
    components = fname.split('_')
    filename = path+fname
    data = mrc_raw_import(filename)
    subject_id = components[4]
    eyes = components[3]
    visit = components[6]
    group = components[5]
    remica = components[0]
    task = components[2]
    print fname
    print subject_id+eyes+visit+group
    output_file = output_directory+'Subject'+subject_id
    condition = visit+'/'+task+'/'+eyes+'/magnetometer/'+remica
    write_to_HDF5(data['magnetometer'],output_file, condition, 250.0, bands=bands,\
            window=window, taps=taps,\
            group_name='GSK'+group, species='human', location='MRC', number_in_group=int(subject_id))
    condition = visit+'/'+task+'/'+eyes+'/gradiometer/'+remica
    write_to_HDF5(data['gradiometer'],output_file, condition, 250.0, bands=bands,\
            window=window, taps=taps,\
            group_name='GSK'+group, species='human', location='MRC', number_in_group=int(subject_id))
