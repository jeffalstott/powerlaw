import criticality
import h5py
from numpy import arange
import os
import sqlite3

bins = [1,  3, 5, 7, 9]]
percentiles = [.9, .95, .99, .999]
event_methods = ['amplitude']
cascade_methods = ['grid']
subsamples = [('all', 'all')]


visits = ['2, 3']
tasks = ['rest']
eyes = ['open']
sensors = ['gradiometer']

group_name ='GSK1'
data_path = '/work/imagingA/jja34/MEG_Study/Data/'+group_name
database = '/work/imagingA/jja34/Results'
filter_type = 'FIR'
taps = 513
window = 'blackmanharris'


dirList=os.listdir(data_path)

for fname in dirList:
    file = data_path+'/'+fname
    number_in_group = int(fname[7:10])
    conn = sqlite3.connect(database)
    values = ('human', group_name, number_in_group)
    ids = conn.execute("SELECT subject_id FROM Subjects WHERE species=? AND group_name=? and number_in_group=?",\
            values).fetchall()
    if len(ids)==0:
        cur = conn.execute("INSERT INTO Subjects (species, group_name, number_in_group) values (?, ?, ?)",\
                values)
        subject_id = cur.lastrowid
    else:
        subject_id = ids[0][0]
    conn.commit()
    conn.close()

    print file
    f = h5py.File(file)
    conditions = [(v,t,e,s) for v in visits for t in tasks for e in eyes for s in sensors] 
    for visit, task, eye, sensor in conditions:
        base = visit+'/'+task+'/'+eye+'/'+sensor
        base_filtered = base+'/filter_'+filter_type+'_'+str(taps)+'_'+window
        #If this particular set of conditions doesn't exist in for this subject, just continue to the next set of conditions
        try:
            f[base_filtered]
        except KeyError:
            continue
        print base

        duration = f[base+'/raw/displacement'].shape[1]

        conn = sqlite3.connect(database)
        cur = conn.execute("select task_id from Tasks where type=? and eyes=?", (task, eye)).fetchall();
        task_id = cur[0][0]

        cur = conn.execute("select sensor_id from Sensors where location='MRC' and sensor_type=?", (sensor,)).fetchall();
        sensor_id = cur[0][0]

        values = ('MRC', subject_id, visit, 50, 'none', 'rested', task_id)
        ids = conn.execute("SELECT experiment_id FROM Experiments WHERE location=? AND subject_id=? AND \
                visit_number=? AND mains=? AND drug=? AND rest=? AND task_id=?", values).fetchall()
        if len(ids)==0:
            cur = conn.execute("INSERT INTO Experiments (location, subject_id, visit_number, mains, drug, rest, task_id)\
                    values (?, ?, ?, ?, ?, ?, ?)", values)
            experiment_id = cur.lastrowid
        else:
            experiment_id=ids[0][0]

        values = (experiment_id, sensor_id, duration)
        ids = conn.execute('SELECT recording_id FROM Recordings_Raw WHERE experiment_id=? AND sensor_id=? \
                AND duration=?', values).fetchall()
        if len(ids)==0:
            cur = conn.execute("INSERT INTO Recordings_Raw (experiment_id, sensor_id, duration) values (?,?,?)",\
                    values)
            recording_id = cur.lastrowid
        else:
            recording_id=ids[0][0]
        conn.commit()
        conn.close()

        for band in list(f[base_filtered]):
	    print band
            data = f[base_filtered+'/'+band]
            conn = sqlite3.connect(database)
	    band_range = data.attrs['frequency_range']
	    if band_range.shape[0]==1:
		band_min=0.
		band_max=band_range[0]
	    else:
		band_min=band_range[0]
		band_max=band_range[1]
            values = (recording_id, filter_type, taps-1, window, band, \
                    band_min, band_max, \
                    data['displacement'].shape[1], 0, 0)
            ids = conn.execute("SELECT filter_id FROM Recordings_Filtered WHERE recording_id=? AND filter_type=?\
                    AND poles=? AND window=? AND band_name=? AND band_min=? AND band_max=? AND duration=?\
                    AND notch=? AND phase_shuffled=?", values).fetchall()
            if len(ids)==0:
                cur = conn.execute("INSERT INTO Recordings_Filtered (recording_id, filter_type, poles, window,\
                        band_name, band_min, band_max, duration, notch, phase_shuffled) values (?,?,?,?,?,?,?,?,?,? )",\
                        values)
                filter_id = cur.lastrowid
            else:
                filter_id = ids[0][0]
            conn.commit()
            conn.close()

            criticality.avalanche_analyses(data, bins, percentiles, event_methods, cascade_methods, subsamples,\
                    write_to_database=database, filter_id=filter_id)
