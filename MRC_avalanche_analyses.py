import criticality
import h5py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import database_classes as dc

bins = [1, 2, 4]
percentiles = [.9921875, .984375, .96875]
event_methods = ['amplitude']
cascade_methods = ['grid']
spatial_samples = [('all', 'all')]
temporal_samples = [('all', 'all')]


visits = ['2', '3']
tasks = ['rest']
eyes = ['open']
sensors = ['gradiometer']

group_name ='GSK1'
species = 'human'
location='MRC'
data_path = '/work/imagingA/jja34/Data/MRC/'+group_name
#database = '/work/imagingA/jja34/Results'
database = 'mysql://jja34:r38bf2u9@iron/JeffAnalysis'
#database = 'sqlite:///:memory:'
filter_type = 'FIR'
taps = 513
window = 'blackmanharris'


engine = create_engine(database, echo=False)
dc.Base.metadata.create_all(engine) 
Session = sessionmaker(bind=engine)


dirList=os.listdir(data_path)
for fname in dirList:
    file = data_path+'/'+fname
    number_in_group = int(fname[7:10])
    session = Session()

    subject = session.query(dc.Subject).\
            filter_by(species=species, group_name=group_name, number_in_group=number_in_group).first()
    if not subject:
        subject = dc.Subject(species=species, group_name=group_name, number_in_group=number_in_group)
        session.add(subject)
        session.commit()

    print file
    f = h5py.File(file)
    conditions = [(v,t,e,s) for v in visits for t in tasks for e in eyes for s in sensors] 
    for visit, task_type, eye, sensor_type in conditions:
        base = visit+'/'+task_type+'/'+eye+'/'+sensor_type
        base_filtered = base+'/filter_'+filter_type+'_'+str(taps)+'_'+window
        #If this particular set of conditions doesn't exist for this subject, just continue to the next set of conditions
        try:
            f[base_filtered]
        except KeyError:
            continue
        print base

        duration = f[base+'/raw/displacement'].shape[1]

        task = session.query(dc.Task).\
                filter_by(type=task_type, eyes=eye).first()
        if not task:
            print('Task not found!')
            break

        sensor = session.query(dc.Sensor).\
                filter_by(location=location, sensor_type=sensor_type).first()
        if not sensor:
            print('Sensor not found!')
            break
        
        experiment = session.query(dc.Experiment).\
                filter_by(location=location, subject_id=subject.id, visit_number=visit, mains=50, drug='none',\
                rest='rested', task_id=task.id).first()
        if not experiment:
            experiment = dc.Experiment(location=location, subject_id=subject.id, visit_number=visit, mains=50, drug='none',\
                rest='rested', task_id=task.id)
            session.add(experiment)
            session.commit()

        recording = session.query(dc.Recording).\
                filter_by(experiment_id=experiment.id, sensor_id=sensor.id, duration=duration).first()
        if not recording:
            recording = dc.Recording(experiment_id=experiment.id, sensor_id=sensor.id, duration=duration,\
                    subject_id = subject.id, task_id=task.id)
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

            filter = session.query(dc.Filter).\
                    filter_by(recording_id=recording.id, filter_type=filter_type, poles=taps-1, window=window,\
                    band_name=band, band_min=band_min, band_max=band_max, duration=data['displacement'].shape[1],\
                    notch=False,phase_shuffled=False).first()
            if not filter:
                filter = dc.Filter(\
                    recording_id=recording.id, filter_type=filter_type, poles=taps-1, window=window,\
                    band_name=band, band_min=band_min, band_max=band_max, duration=data['displacement'].shape[1],\
                    notch=False,phase_shuffled=False,\
                    subject_id = subject.id, task_id=task.id, experiment_id=experiment.id, sensor_id=sensor.id)

                session.add(filter)
                session.commit()

            criticality.avalanche_analyses(data, \
                    bins=bins, percentiles=percentiles, event_methods=event_methods, cascade_methods=cascade_methods, \
                    spatial_samples=spatial_samples, temporal_samples=temporal_samples,\
                    write_to_database=database, filter_id=filter.id,\
                    subject_id=subject.id, task_id=task.id, experiment_id=experiment.id,\
                    sensor_id=sensor.id, recording_id=recording.id,\
                    verbose=True)
