import avalanches
import h5py
import os

import Helix_database as db
session = db.Session()
cluster=True
analyses_directory = '/home/alstottj/biowulf/analyses/'
swarms_directory = '/home/alstottj/biowulf/swarms/'
python_location= '/usr/local/Python/2.7.2/bin/python'

time_scales = [1, 2, 3, 4, 5, 6, 7, 8, 16, 32]
threshold_mode = 'SD'
threshold_levels = [3]
threshold_directions = ['both']
given_xmin_xmax = [(None, None), (1, None), (1, 'channels')]
event_signals = ['amplitude', 'displacement']
event_detections = ['local_extrema', 'local', 'excursion_extrema']
cascade_methods = ['grid']
spatial_samples = [('all', 'all')]
temporal_samples = [('all', 'all')]


tasks = ['rest']
eyes = ['shut', 'open']
sensors = ['gradiometer', 'magnetometer']
sampling_rate = 1000.0

data_path = '/data/alstottj/RIKEN/'
filter_type = 'FIR'
taps = 513
window = 'blackmanharris'
transd = True
mains = 50

monkeys =[('A',5), ('K1', 4), ('K2',3)]
tasks = [('food_tracking0', 3), ('food_tracking1', 3), ('food_tracking2', 3), ('food_tracking3', 3),\
        ('food_tracking4', 3), ('visual_grating', 6), ('emotional_movie', 7), ('rest', 5), ('anesthesia', 4)]

rem=False
rest='rested'
drug='none'
visit=0

dirList=os.listdir(data_path)
for fname in dirList:
    file = data_path+fname
    f = h5py.File(file)
    group_name = f.attrs['group_name'] 
    number_in_group = f.attrs['number_in_group']
    species = f.attrs['species']
    location = f.attrs['location']

    subject = session.query(db.Subject).\
            filter_by(species=species, group_name=group_name, number_in_group=number_in_group).first()
    if not subject:
        subject = db.Subject(species=species, group_name=group_name, number_in_group=number_in_group)
        session.add(subject)
        session.commit()

    print file

    conditions = [(t) for t in tasks] 
    for t in conditions:
        base = t
        base_filtered = base+'/filter_'+filter_type+'_'+str(taps)+'_'+window
        #If this particular set of conditions doesn't exist for this subject, just continue to the next set of conditions
        try:
            f[base_filtered]
        except KeyError:
            continue
        print base

        duration = f[base+'/raw/displacement'].shape[1]

        task = session.query(db.Task).\
                filter_by(type=task_type).first()
        if not task:
            print('Task not found!')
            break

        sensor = session.query(db.Sensor).\
                filter_by(location=location, sensor_type=sensor_type).first()
        if not sensor:
            print('Sensor not found!')
            break
        
        experiment = session.query(db.Experiment).\
                filter_by(location=location, subject_id=subject.id, visit_number=visit, mains=mains, drug=drug,\
                rest=rest, task_id=task.id).first()
        if not experiment:
            experiment = db.Experiment(location=location, subject_id=subject.id, visit_number=visit, mains=mains, drug=drug,\
                rest=rest, task_id=task.id)
            session.add(experiment)
            session.commit()


        recording = session.query(db.Recording).\
                filter_by(experiment_id=experiment.id, sensor_id=sensor.id, duration=duration, \
                subject_id = subject.id, task_id=task.id,\
                sampling_rate=sampling_rate, eye_movement_removed=rem, transd=transd).first()
        if not recording:
            recording = db.Recording(experiment_id=experiment.id, sensor_id=sensor.id, duration=duration,\
                    subject_id = subject.id, task_id=task.id,\
                    sampling_rate=sampling_rate, eye_movement_removed=rem, transd=transd)
            session.add(recording)
            session.commit()

        for band in list(f[base_filtered]):
            print band
            data = f[base_filtered+'/'+band]
            band_range = data.attrs['frequency_range']
            if band_range.shape[0]==1:
                band_min=0.
                band_max=band_range[0]
            else:
                band_min=band_range[0]
                band_max=band_range[1]

            filter = session.query(db.Filter).\
                    filter_by(recording_id=recording.id, filter_type=filter_type, poles=taps-1, window=window,\
                    band_name=band, band_min=band_min, band_max=band_max, duration=data['displacement'].shape[1],\
                    notch=False,phase_shuffled=False).first()
            if not filter:
                filter = db.Filter(\
                    recording_id=recording.id, filter_type=filter_type, poles=taps-1, window=window,\
                    band_name=band, band_min=band_min, band_max=band_max, duration=data['displacement'].shape[1],\
                    notch=False,phase_shuffled=False,\
                    subject_id = subject.id, task_id=task.id, experiment_id=experiment.id, sensor_id=sensor.id)

                session.add(filter)
                session.commit()

            avalanches.avalanche_analyses(f.file.filename, HDF5_group=base_filtered+'/'+band,\
                    threshold_mode=threshold_mode, threshold_levels=threshold_levels, threshold_directions=threshold_directions,\
                    event_signals=event_signals, event_detections=event_detections,\
                    time_scales=time_scales, cascade_methods=cascade_methods,\
                    given_xmin_xmax=given_xmin_xmax,\
                    spatial_samples=spatial_samples, temporal_samples=temporal_samples,\
                    session=session, database_url=db.database_url,\
                    subject_id=subject.id, task_id=task.id, experiment_id=experiment.id,\
                    sensor_id=sensor.id, recording_id=recording.id, filter_id=filter.id,\
                    cluster=cluster, swarms_directory=swarms_directory, analyses_directory=analyses_directory,\
                    python_location=python_location,\
                    verbose=True)
session.close()
