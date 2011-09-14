import criticality
import h5py
import sqlite3

bins = [1, 2, 4]
percentiles = [.9921875, .984375, .96875]
event_methods = ['amplitude']
cascade_methods = ['grid']
subsamples = [('all', 'all')]

data_path = '/work/imagingA/jja34/ECOG_Study/Data'
database = '/work/imagingA/jja34/Results'
filter_type = 'FIR'
taps = 513
window = 'blackmanharris'

monkeys =[('A',5), ('K1', 4), ('K2',3)]
tasks = [('food_tracking0', 3), ('food_tracking1', 3), ('food_tracking2', 3), ('food_tracking3', 3),\
        ('food_tracking4', 3), ('visual_grating', 6), ('emotional_movie', 7), ('rest', 5), ('anesthesia', 4)]

for name, sensor_id in monkeys:
    file = data_path+'/Monkey_'+name+'.hdf5'
    conn = sqlite3.connect(database)
    values = ('monkey', 'RIKEN', name)
    ids = conn.execute("SELECT subject_id FROM Subjects WHERE species=? AND group_name=? and name=?",\
            values).fetchall()
    if len(ids)==0:
        cur = conn.execute("INSERT INTO Subjects (species, group_name, name) values (?, ?, ?)",\
                values)
        subject_id = cur.lastrowid
    else:
        subject_id = ids[0][0]
    conn.commit()
    conn.close()

    print file
    f = h5py.File(file)
    for task_name, task_id in tasks:
        base = task_name
        base_filtered = base+'/filter_'+filter_type+'_'+str(taps)+'_'+window
        #If this particular set of conditions doesn't exist in for this subject, just continue to the next set of conditions
        try:
            f[base_filtered]
        except KeyError:
            continue
        print base

        duration = f[base+'/raw/displacement'].shape[1]

        conn = sqlite3.connect(database)

        values = ('RIKEN', subject_id, 50, task_id)
        ids = conn.execute("SELECT experiment_id FROM Experiments WHERE location=? AND subject_id=? \
                AND mains=? AND task_id=?", values).fetchall()
        if len(ids)==0:
            cur = conn.execute("INSERT INTO Experiments (location, subject_id, mains, task_id)\
                    values (?, ?, ?, ?)", values)
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
