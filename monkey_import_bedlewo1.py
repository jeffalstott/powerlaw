from scipy.io import loadmat
file_base = '/home/jalstott/Monkey_ECoG/Rest/ECoG_ch'
variable_base = 'ECoGData_ch'

monkey_data = empty((128,2062570),dtype=int16)

for i in range(128):
	f = str.format('{0}{1}.mat', file_base, i+1)
	v = str.format('{0}{1}', variable_base, i+1)
	monkey_data[i,:] = loadmat(f)[v]

