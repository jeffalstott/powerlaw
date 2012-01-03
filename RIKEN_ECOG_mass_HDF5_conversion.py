import os
from input_processing import riken_import, write_to_HDF5

#path = '/work/imaging8/jja34/ECoG_Study/ECoG_Data/data'
path = '/data/alstottj/RIKEN/For_Filtering/data'
output_path = '/data/alstottj/RIKEN/For_Analysis/'

dirList=os.listdir(path)

bands = ('delta', 'theta', 'alpha', 'beta', 'raw', 'gamma', 'high-gamma', 'broad')
#bands = ('delta', 'theta', 'alpha', 'beta')
window='hamming'
taps=25

tasks = {'FTT': 'food_tracking', 'EMT': 'emotional_movie', 'VGT': 'visual_grating', \
        'SCT': 'social_competition'}

A_counter = 0
K1_counter = 0
K2_counter = 0
for dirname in dirList:
    print dirname
    filename = path+'/'+dirname+'/'
    if 'ECoG_ch1.mat' not in os.listdir(filename):
        print("Wrong data format")
        continue
    data = riken_import(filename)
    components = dirname.split('_')
    name = components[2]
    date = components[0][0:7]
    if components[1] not in tasks:
        continue
    task = tasks[components[1]]
    if name=='A' and task=='food_tracking':
        task = task+str(A_counter)
        A_counter = A_counter+1
    if name=='K1' and task=='food_tracking':
        task = task+str(K1_counter)
        K1_counter = K1_counter+1
    if name=='K2' and task=='visual_grating':
        task = task+str(K2_counter)
        K2_counter = K2_counter+1

    #output_file = '/work/imaging8/jja34/ECoG_Study/ECoG_Data/Monkey_'+name
    output_file = output_path+'Monkey_'+name
    write_to_HDF5(data,output_file, task, 1000.0, bands=bands,\
            window=window, taps=taps,\
            group_name='RIKEN', species='monkey', location='RIKEN', number_in_group=name, name=name, date=date)
