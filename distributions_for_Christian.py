import avalanches
import h5py
import os

output_file_base = '/home/alstottj/distributions_for_Christian_'
visits = [2, 3]
tasks = ['rest']
eyes = ['shut', 'open']
sensors = ['gradiometer']
remicas = ['raw']
sampling_rate = 250.0

data_path = '/data/alstottj/MRC/'
filter_type = 'FIR'
taps = '513'
window = 'blackmanharris'
transd = True
mains = 50

dirList=os.listdir(data_path)
for fname in dirList:
    print fname
    file = data_path+fname
    f = h5py.File(file)
    group_name = f.attrs['group_name'] 
    number_in_group = f.attrs['number_in_group']
    
    if group_name=='GSK1':
        if not '2' in f.keys() or not '3' in f.keys():
            continue
        drug = ''
        output_file = open(output_file_base+group_name+drug+'.txt', 'a')

        data = f['2/rest/open/gradiometer/raw/filter_'+filter_type+'_'+taps+'_'+window+'/broad/']
        metrics = avalanches.avalanche_analysis(data)
        d = metrics['size_events']
        output_file.writelines([','.join(map(str, d))])
        output_file.write('\n')

        data = f['3/rest/open/gradiometer/raw/filter_'+filter_type+'_'+taps+'_'+window+'/broad/']
        metrics = avalanches.avalanche_analysis(data)
        d = metrics['size_events']
        output_file.write(', '.join(map(str, d)))
        output_file.write('\n')
        output_file.write('\n')
        output_file.close()

    elif group_name=='GSK2' and number_in_group in [21, 25, 30, 35, 37, 44, 48, 137, 148]:
        if not '2' in f.keys() or not '3' in f.keys():
            continue
        drug = 'placebo'
        output_file = open(output_file_base+group_name+drug+'.txt', 'a')

        data = f['3/rest/open/gradiometer/raw/filter_'+filter_type+'_'+taps+'_'+window+'/broad/']
        metrics = avalanches.avalanche_analysis(data)
        d = metrics['size_events']
        output_file.write(', '.join(map(str, d)))
        output_file.write('\n')

        data = f['2/rest/open/gradiometer/raw/filter_'+filter_type+'_'+taps+'_'+window+'/broad/']
        metrics = avalanches.avalanche_analysis(data)
        d = metrics['size_events']
        output_file.write(', '.join(map(str, d)))
        output_file.write('\n')
        output_file.write('\n')
        output_file.close()
    elif group_name=='GSK2' and number_in_group in [22, 28, 32, 34, 39, 42, 45, 49, 50]:
        if not '2' in f.keys() or not '3' in f.keys():
            continue
        drug = 'placebo'
        output_file = open(output_file_base+group_name+drug+'.txt', 'a')

        data = f['2/rest/open/gradiometer/raw/filter_'+filter_type+'_'+taps+'_'+window+'/broad/']
        metrics = avalanches.avalanche_analysis(data)
        d = metrics['size_events']
        output_file.write(', '.join(map(str, d)))
        output_file.write('\n')

        data = f['3/rest/open/gradiometer/raw/filter_'+filter_type+'_'+taps+'_'+window+'/broad/']
        metrics = avalanches.avalanche_analysis(data)
        d = metrics['size_events']
        output_file.write(', '.join(map(str, d)))
        output_file.write('\n')
        output_file.write('\n')
        output_file.close()
    elif group_name=='GSK2' and number_in_group in [23, 27,33, 38,46,143]:
        if not '2' in f.keys() or not '3' in f.keys():
            continue
        drug = 'donepezil'
        output_file = open(output_file_base+group_name+drug+'.txt', 'a')

        data = f['3/rest/open/gradiometer/raw/filter_'+filter_type+'_'+taps+'_'+window+'/broad/']
        metrics = avalanches.avalanche_analysis(data)
        d = metrics['size_events']
        output_file.write(', '.join(map(str, d)))
        output_file.write('\n')

        data = f['2/rest/open/gradiometer/raw/filter_'+filter_type+'_'+taps+'_'+window+'/broad/']
        metrics = avalanches.avalanche_analysis(data)
        d = metrics['size_events']
        output_file.write(', '.join(map(str, d)))
        output_file.write('\n')
        output_file.write('\n')
        output_file.close()
    elif group_name=='GSK2' and number_in_group in [24, 26, 31, 36, 40, 41, 231, 149]:
        if not '2' in f.keys() or not '3' in f.keys():
            continue
        drug = 'donepezil'
        output_file = open(output_file_base+group_name+drug+'.txt', 'a')

        data = f['2/rest/open/gradiometer/raw/filter_'+filter_type+'_'+taps+'_'+window+'/broad/']
        metrics = avalanches.avalanche_analysis(data)
        d = metrics['size_events']
        output_file.write(', '.join(map(str, d)))
        output_file.write('\n')

        data = f['3/rest/open/gradiometer/raw/filter_'+filter_type+'_'+taps+'_'+window+'/broad/']
        metrics = avalanches.avalanche_analysis(data)
        d = metrics['size_events']
        output_file.write(', '.join(map(str, d)))
        output_file.write('\n')
        output_file.write('\n')
        output_file.close()
    else: 
        print("Couldn't find this subject!")
