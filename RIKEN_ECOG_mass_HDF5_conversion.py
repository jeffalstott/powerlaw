import os
from input_processing import riken_import, write_to_HDF5

#path = '/work/imaging8/jja34/ECoG_Study/ECoG_Data/data'
path = '/data/alstottj/RIKEN/For_Filtering/'
output_path = '/data/alstottj/RIKEN/For_Analysis/'

dirList=os.listdir(path)

bands = ('delta', 'theta', 'alpha', 'beta', 'raw', 'gamma', 'high-gamma', 'broad')
window='hamming'
taps=25
downsample=100.0

tasks = {'FTT': 'food_tracking', 'EMT': 'emotional_movie', 'VGT': 'visual_grating', \
        'SCT': 'social_competition', 'ST': 'sleep_task'}

A_counter = 0
K1_counter = 0
K2_counter = 0
for dirname in dirList:
    print dirname
    filename = path+dirname+'/'
    components = dirname.split('_')
    name = components[2]
    output_file = output_path+'Monkey_'+name

    date = components[0][0:7]
    if components[1] not in tasks:
        continue
    task = tasks[components[1]]
    if name=='A' and task=='food_tracking':
        continue
        task = task+str(A_counter)
        A_counter = A_counter+1
    if name=='K1' and task=='food_tracking':
        continue
        task = task+str(K1_counter)
        K1_counter = K1_counter+1
    if name=='K2' and task=='visual_grating':
        task = task+str(K2_counter)
        K2_counter = K2_counter+1
    if name=='K2' and task=='sleep_task':
        data = riken_import(filename)

        write_to_HDF5(data,output_file, task, 1000.0, bands=bands,\
                window=window, taps=taps,\
                downsample=downsample,
                group_name='RIKEN', species='monkey', location='RIKEN',\
                        number_in_group=name, name=name, date=date)
        write_to_HDF5(data[:,:600000],output_file, 'rest', 1000.0, bands=bands,\
                window=window, taps=taps,\
                downsample=downsample,
                group_name='RIKEN', species='monkey', location='RIKEN',\
                        number_in_group=name, name=name, date=date)
        write_to_HDF5(data[:,-600000:],output_file, 'anesthetized', 1000.0, bands=bands,\
                window=window, taps=taps,\
                downsample=downsample,
                group_name='RIKEN', species='monkey', location='RIKEN',\
                        number_in_group=name, name=name, date=date)
        write_to_HDF5(data[:,600001:-600001],output_file, 'sleep_wake_transition', 1000.0, bands=bands,\
                window=window, taps=taps,\
                downsample=downsample,
                group_name='RIKEN', species='monkey', location='RIKEN',\
                        number_in_group=name, name=name, date=date)
        continue

    data = riken_import(filename)

    write_to_HDF5(data,output_file, task, 1000.0, bands=bands,\
            window=window, taps=taps,\
            downsample=downsample,
            group_name='RIKEN', species='monkey', location='RIKEN',\
                    number_in_group=name, name=name, date=date)
