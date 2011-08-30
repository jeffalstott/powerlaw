import os
from input_processing import riken_import, write_to_HDF5

path = '/work/imaging8/jja34/ECoG_Study/ECoG_Data/data'
#path = '/home/alstottj/data'
dirList=os.listdir(path)

#bands = ('delta', 'theta', 'alpha', 'beta')
bands = ('raw', 'gamma', 'high-gamma')

tasks = {'FTT': 'food_tracking', 'EMT': 'emotional_movie', 'VGT': 'visual_grating', \
        'SCT': 'social_competition'}

A_counter = -1
K1_counter = -1
for dirname in dirList:
    data = riken_import(dirname)
    print dirname
    components = dirname.split('_')
    name = components[2]
    task = tasks[components[1]]
    if name=='A':
        A_counter = A_counter+1
        task = task+str(A_counter)
    if name=='K1':
        K1_counter = K1_counter+1
        task = task+str(K1_counter)

    output_file = '/work/imaging8/jja34/ECoG_Study/ECoG_Data/Monkey_'+name
    #output_file = '/scratch/alstottj/Monkey_'+name
    write_to_HDF5(data,output_file, task, 1000.0, bands=bands)


