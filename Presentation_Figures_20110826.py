import h5py
import criticality
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

figure_directory = '/home/jja34/public_html/Figures/' 
data_directory = '/work/imaging8/jja34/ECoG_Study/ECoG_Data/'

bands = ('beta', 'alpha', 'theta', 'delta')
method = 'amplitude_aucs'


filter = 'filter_FIR_513_blackmanharris'
task = 'food_tracking'
for band in bands:
    plt.subplot(1,2,1)
    m = 'A'
    b=3
    i = 5
    f = h5py.File(data_directory+'Monkey_'+m+'.hdf5')
    for task_ind in range(i):
        p=.99
        data = f['food_tracking'+task_ind+'/'+filter+'/'+band]
        d = criticality.avalanche_analysis(data, bin_width=b, percentile=p)
        X = d['size_'+method]
        statistics.hist_log(X, X.max(), X.min())
    plt.xlabel('Size ('+method+')', fontsize='xx-large')
    plt.ylabel('P(Size)', fontsize='xx-large')
    plt.title(band+' band, Monkey '+m, fontsize='xx-large')
    plt.legend(('Trial 1', 'Trial 2', 'Trial 3', 'Trial 4', 'Trial 5'))
    f.close()
            
    plt.subplot(1,2,2)
    m = 'K1'
    b=1
    i = 3
    f = h5py.File(data_directory+'Monkey_'+m+'.hdf5')
    for task_ind in range(i):
        p=.99
        data = f['food_tracking'+task_ind+'/'+filter+'/'+band]
        d = criticality.avalanche_analysis(data, bin_width=b, percentile=p)
        X = d['size_'+method]
        statistics.hist_log(X, X.max(), X.min())
    plt.xlabel('Size ('+method+')', fontsize='xx-large')
    plt.ylabel('P(Size)', fontsize='xx-large')
    plt.title(band+' band, Monkey '+m, fontsize='xx-large')
    plt.legend(('Trial 1', 'Trial 2', 'Trial 3'))
    plt.savefig(figure_directory+'Presentation_Figures_20110826_2_'+band)
    f.close()

filter = 'filter_version0'
tasks = ('rest', 'anesthesia')
for i in range(4):
    plt.subplot(1,2,i+1)
    band = bands[i]
    m = 'K2'
    b=1
    for task in tasks:
        p=.992
        f = h5py.File(data_directory+'Monkey_'+m+'.hdf5')
        data = f[task+'/'+filter+'/'+band]
        d = criticality.avalanche_analysis(data, bin_width=b, percentile=p)
        X = d['size_'+method]
        statistics.hist_log(X, X.max(), X.min())
    plt.xlabel('Size ('+method+')', fontsize='xx-large')
    plt.ylabel('P(Size)', fontsize='xx-large')
    plt.title(band+' band, Monkey '+m, fontsize='xx-large')
    plt.legend(('rest', 'anesthesia'))
            
plt.savefig(figure_directory+'Presentation_Figures_20110826_1')
f.close()
