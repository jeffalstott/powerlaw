from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database_classes import *
import database_classes

database_url = 'mysql://alstottj:RusduOv4@biobase/alstottj'

engine = create_engine(database_url, echo=False)
#Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

def compare(*args, **kwargs):
    """compare does things"""
    session = Session()

    data = session.query(*args).\
        join(Fit_Association).\
        join(Avalanche, Avalanche.id==Fit_Association.id).\
        join(Filter, Filter.id==Fit.filter_id).\
        join(Recording, Recording.id==Fit.recording_id).\
        join(Experiment, Experiment.id==Fit.experiment_id).\
        join(Sensor, Sensor.id==Fit.sensor_id).\
        join(Subject, Subject.id==Fit.subject_id).\
        join(Task, Task.id==Fit.task_id)

    filters = {
        'Sensor.sensor_type': 'gradiometer',\
        'Task.eyes': 'open',\
        'Experiment.visit_number': None,\
        'Filter.band_name': 'broad',\
        'Filter.downsampled_rate': '1000',\
        'Subject.group_name': None,\
        'Avalanche.spatial_sample': 'all',\
        'Avalanche.temporal_sample': 'all',\
        'Avalanche.threshold_mode': 'SD',\
        'Avalanche.threshold_level': 3,\
        'Avalanche.threshold_direction': 'both',\
        'Avalanche.event_signal': 'displacement',\
        'Avalanche.event_detection': 'local_extrema',\
        'Avalanche.cascade_method': 'grid',\
        'Fit.analysis_type':  'avalanches',\
        'Fit.variable':  'size_events',\
        'Fit.distribution':  'power_law',\
        'Fit.fixed_xmin':  True,\
        'Fit.xmin': 1,\
        'Fit.fixed_xmax':  True,\
        'Fit.xmax': 204,\
        'Avalanche.time_scale': 2,\
        }

    filters.update(kwargs)
    print filters

    for key in filters:
        if filters[key]==None:
            continue
        table, variable = key.split('.')
        data = data.filter(getattr(getattr(database_classes,table),variable)==filters[key])

    d = data.all()
    session.close()
    session.bind.dispose()
    types = []
    for a in range(len(args)):
        name = str(args[a]).split('.')[-1]
        t = type(d[0][a])
        if t==str:
            t = '|S100'
        types.append((name, t))
    
    from numpy import array
    return array(d, dtype = types) 
