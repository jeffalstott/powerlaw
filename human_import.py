def fieldtrip_import(filename):
    from scipy.io import loadmat
    from numpy import where
    matfile = loadmat(filename)
    data = matfile['D']['trial'][0][0][0][0]
    activity_starts_index = where(data.sum(0)!=0)[0][0]
    data = data[:, activity_starts_index:]


    output = {}
    output['magnetometer'] = data[0:102]
    output['gradiometer'] = data[102:306]
    return output
