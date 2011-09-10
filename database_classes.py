from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

from sqlalchemy import Column, Float, Integer, String, ForeignKey
from sqlalchemy.orm import relationship, backref


class Task(Base):
    __tablename__ = 'Tasks'
    
    id = Column(Integer, primary_key=True)
    type = Column(String)
    description = Column(String)

class Subject(Base):
    __tablename__ = 'Subjects'

    id = Column(Integer, primary_key=True)
    species = Column(String)
    name = Column(String)
    group_name = Column(String)
    number_in_group = Column(String)

class Sensor(Base):
    __tablename__ = 'Sensors'

    id = Column(Integer, primary_key=True)
    location = Column(String)
    sensor_type = Column(String)
    sensor_count = Column(Integer)
    sensors_locations_file = Column(String)
    sensors_spacing = Column(Float)

class Experiment(Base):
    __tablename__ = 'Experiments'

    id = Column(Integer, primary_key=True)
    location = Column(String)
    date = Column(String)
    visit_number = Column(Integer)
    mains = Column(Integer)
    drug = Column(String)
    rest = Column(String)

    subject_id = Column(Integer, ForeignKey('Subjects.id'))
    task_id = Column(Integer, ForeignKey('Tasks.id'))

    subject = relationship(Subject, backref=backref('experiments', order_by=id))
    task = relationship(Task, backref=backref('experiments', order_by=id))

class Recording(Base):
    __tablename__ = 'Recordings'

    id = Column(Integer, primary_key=True)
    duration = Column(Float)
    sampling_rate = Column(Float)

    subject_id = Column(Integer, ForeignKey('Subjects.id'))
    task_id = Column(Integer, ForeignKey('Tasks.id'))
    experiment_id = Column(Integer, ForeignKey('Experiments.id'))
    sensor_id = Column(Integer, ForeignKey('Sensors.id'))

    subject = relationship(Subject, backref=backref('recordings', order_by=id))
    task = relationship(Task, backref=backref('recordings', order_by=id))
    experiment = relationship(Experiment, backref=backref('recordings', order_by=id))
    sensor = relationship(Sensor, backref=backref('recordings', order_by=id))

class Filter(Base):
    __tablename__ = 'Filters'

    id = Column(Integer, primary_key=True)
    filter_type = Column(String)
    poles = Column(Integer)
    window = Column(Integer)
    band_name = Column(String)
    band_min = Column(Float)
    band_max = Column(Float)
    duration = Column(Float)
    notch = Column(Integer)
    phase_shuffled = Column(Integer)

    subject_id = Column(Integer, ForeignKey('Subjects.id'))
    task_id = Column(Integer, ForeignKey('Tasks.id'))
    experiment_id = Column(Integer, ForeignKey('Experiments.id'))
    recording_id = Column(Integer, ForeignKey('Recordings.id'))
    sensor_id = Column(Integer, ForeignKey('Sensors.id'))

    subject = relationship(Subject, backref=backref('filters', order_by=id))
    task = relationship(Task, backref=backref('filters', order_by=id))
    experiment = relationship(Experiment, backref=backref('filters', order_by=id))
    sensor = relationship(Sensor, backref=backref('filters', order_by=id))
    recording = relationship(Recording, backref=backref('filters', order_by=id))

class Avalanche_Analysis(Base):
    __tablename__ = 'Avalanche_Analyses'

    id = Column(Integer, primary_key=True)
    subsample = Column(String)
    threshold_mode = Column(String)
    threshold_level = Column(Float)
    time_scale = Column(Float)
    event_method = Column(String)
    cascade_method = Column(String)
    interevent_interval = Column(Float)
    sigma_events = Column(Float)
    sigma_displacements = Column(Float)
    sigma_amplitudes = Column(Float)
    sigma_amplitude_aucs = Column(Float)
    t_ratio_displacements_slope = Column(Float)
    t_ratio_displacements_R = Column(Float)
    t_ratio_displacements_p = Column(Float)
    t_ratio_amplitudes_slope = Column(Float)
    t_ratio_amplitudes_R = Column(Float)
    t_ratio_amplitudes_p = Column(Float)
    t_ratio_displacement_aucs_slope = Column(Float)
    t_ratio_displacement_aucs_R = Column(Float)
    t_ratio_displacement_aucs_p = Column(Float)
    t_ratio_amplitude_aucs_slope = Column(Float)
    t_ratio_amplitude_aucs_R = Column(Float)
    t_ratio_amplitude_aucs_p = Column(Float)
    
    subject_id = Column(Integer, ForeignKey('Subjects.id'))
    task_id = Column(Integer, ForeignKey('Tasks.id'))
    experiment_id = Column(Integer, ForeignKey('Experiments.id'))
    recording_id = Column(Integer, ForeignKey('Recordings.id'))
    filter_id = Column(Integer, ForeignKey('Filters.id'))
    sensor_id = Column(Integer, ForeignKey('Sensors.id'))

    subject = relationship(Subject, backref=backref('avalanche_analyses', order_by=id))
    task = relationship(Task, backref=backref('avalanche_analyses', order_by=id))
    experiment = relationship(Experiment, backref=backref('avalanche_analyses', order_by=id))
    sensor = relationship(Sensor, backref=backref('avalanche_analyses', order_by=id))
    recording = relationship(Recording, backref=backref('avalanche_analyses', order_by=id))
    filter = relationship(Filter, backref=backref('avalanche_analyses', order_by=id))

class Distribution_Fit(Base):
    __tablename__ = 'Distribution_Fits'

    id = Column(Integer, primary_key=True)

    analysis_type = Column(String) 
    variable = Column(String) 
    distribution = Column(String) 
    parameter1_name = Column(String) 
    parameter1_value = Column(Float) 
    parameter2_name = Column(String) 
    parameter2_value = Column(Float) 
    parameter3_name = Column(String)
    parameter3_value = Column(Float)
    xmin = Column(Float) 
    xmax = Column(Float)
    loglikelihood = Column(Float) 
    KS = Column(Float)
    p = Column(Float)
    
    subject_id = Column(Integer, ForeignKey('Subjects.id'))
    task_id = Column(Integer, ForeignKey('Tasks.id'))
    experiment_id = Column(Integer, ForeignKey('Experiments.id'))
    recording_id = Column(Integer, ForeignKey('Recordings.id'))
    filter_id = Column(Integer, ForeignKey('Filters.id'))
    analysis_id = Column(Integer, ForeignKey('Avalanche_Analyses.id')) 
    sensor_id = Column(Integer, ForeignKey('Sensors.id'))

    subject = relationship(Subject, backref=backref('distribution_fits', order_by=id))
    task = relationship(Task, backref=backref('distribution_fits', order_by=id))
    experiment = relationship(Experiment, backref=backref('distribution_fits', order_by=id))
    sensor = relationship(Sensor, backref=backref('distribution_fits', order_by=id))
    recording = relationship(Recording, backref=backref('distribution_fits', order_by=id))
    filter = relationship(Filter, backref=backref('distribution_fits', order_by=id))
    analysis = relationship(Avalanche_Analysis, backref=backref('distribution_fits', order_by=id))
