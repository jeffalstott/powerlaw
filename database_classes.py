#Some code, particularly how to join up the Fit table with the Avalanche_Analysis with a polymorphic association, taken from http://techspot.zzzeek.org/files/2007/discriminator_on_association.py
#This polymorphic association was set up in order to allow for future, different kinds of analyses that also would warrant distribution fit analyses
from sqlalchemy import Column, Float, Integer, String, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.associationproxy import association_proxy

class Base(object):
    """Base class which provides automated table name
    and surrogate primary key column.
    
    """
    @declared_attr
    def __tablename__(cls):
        return cls.__name__
    id = Column(Integer, primary_key=True)

Base = declarative_base(cls=Base)


class Task(Base):
    type = Column(String(100))
    description = Column(String(100))
    eyes = Column(String(100))

    def __repr__(self):
        return "<%s(type='%s', description='%s',)>" % (self.__class__.__name__, self.type, self.description)

class Subject(Base):
    species = Column(String(100))
    name = Column(String(100))
    group_name = Column(String(100))
    number_in_group = Column(Integer)

    def __repr__(self):
        return "<%s(species='%s', group='%s')>" % (self.__class__.__name__, self.species, self.group_name)

class Sensor(Base):
    location = Column(String(100))
    sensor_type = Column(String(100))
    sensor_count = Column(Integer)
    sensors_locations_file = Column(String(100))
    sensors_spacing = Column(Float)

    def __repr__(self):
        return "<%s(location='%s', type='%s', count='%s')>" % \
                (self.__class__.__name__, self.location, self.sensor_type, self.sensor_count)

class Experiment(Base):
    location = Column(String(100))
    date = Column(String(100))
    visit_number = Column(Integer)
    mains = Column(Integer)
    drug = Column(String(100))
    rest = Column(String(100))

    subject_id = Column(Integer, ForeignKey('Subject.id'))
    task_id = Column(Integer, ForeignKey('Task.id'))

    subject = relationship(Subject, backref=backref('experiments')) #, order_by=id))
    task = relationship(Task, backref=backref('experiments')) #, order_by=id))

    def __repr__(self):
        return "<%s(subject='%s', visit='%s', task='%s')>" % \
                (self.__class__.__name__, self.subject_id, self.visit_number, self.task_id)

class Task_Performance(Base):
    measure1_name = Column(String(100)) 
    measure1_value = Column(Float) 
    measure2_name = Column(String(100)) 
    measure2_value = Column(Float) 
    measure3_name = Column(String(100))
    measure3_value = Column(Float)
    measure4_name = Column(String(100)) 
    measure4_value = Column(Float) 
    measure5_name = Column(String(100)) 
    measure5_value = Column(Float) 
    measure6_name = Column(String(100))
    measure6_value = Column(Float)
    measure7_name = Column(String(100)) 
    measure7_value = Column(Float) 
    measure8_name = Column(String(100)) 
    measure8_value = Column(Float) 
    measure9_name = Column(String(100))
    measure9_value = Column(Float)
    measure10_name = Column(String(100))
    measure10_value = Column(Float)

    subject_id = Column(Integer, ForeignKey('Subject.id'))
    task_id = Column(Integer, ForeignKey('Task.id'))
    experiment_id = Column(Integer, ForeignKey('Experiment.id'))

    subject = relationship(Subject, backref=backref('task_performances')) #, order_by=id))
    task = relationship(Task, backref=backref('task_performances')) #, order_by=id))
    experiment = relationship(Experiment, backref=backref('task_performances')) #, order_by=id))

    def __repr__(self):
        return "<%s(subject='%s', experiment='%s', task='%s')>" % \
                (self.__class__.__name__, self.subject_id, self.experiment_id, self.task_id)

class Recording(Base):
    duration = Column(Float)
    sampling_rate = Column(Float)
    maxfilter = Column(Boolean)
    transd = Column(Boolean)
    eye_movement_removed = Column(Boolean)

    subject_id = Column(Integer, ForeignKey('Subject.id'))
    task_id = Column(Integer, ForeignKey('Task.id'))
    experiment_id = Column(Integer, ForeignKey('Experiment.id'))
    sensor_id = Column(Integer, ForeignKey('Sensor.id'))

    subject = relationship(Subject, backref=backref('recordings')) #, order_by=id))
    task = relationship(Task, backref=backref('recordings')) #, order_by=id))
    experiment = relationship(Experiment, backref=backref('recordings')) #, order_by=id))
    sensor = relationship(Sensor, backref=backref('recordings')) #, order_by=id))

    def __repr__(self):
        return "<%s(subject='%s', experiment='%s', task='%s', sensor='%s')>" % \
                (self.__class__.__name__, self.subject_id, self.experiment_id, self.task_id, self.sensor_id)

class Filter(Base):
    filter_type = Column(String(100))
    poles = Column(Integer)
    window = Column(String(100))
    band_name = Column(String(100))
    band_min = Column(Float)
    band_max = Column(Float)
    duration = Column(Float)
    notch = Column(Boolean)
    phase_shuffled = Column(Boolean)

    subject_id = Column(Integer, ForeignKey('Subject.id'))
    task_id = Column(Integer, ForeignKey('Task.id'))
    experiment_id = Column(Integer, ForeignKey('Experiment.id'))
    recording_id = Column(Integer, ForeignKey('Recording.id'))
    sensor_id = Column(Integer, ForeignKey('Sensor.id'))

    subject = relationship(Subject, backref=backref('filters')) #, order_by=id))
    task = relationship(Task, backref=backref('filters')) #, order_by=id))
    experiment = relationship(Experiment, backref=backref('filters')) #, order_by=id))
    sensor = relationship(Sensor, backref=backref('filters')) #, order_by=id))
    recording = relationship(Recording, backref=backref('filters')) #, order_by=id))

    def __repr__(self):
        return "<%s(subject='%s', experiment='%s', task='%s', sensor='%s', band='%s')>" % \
                (self.__class__.__name__, self.subject_id, self.experiment_id, self.task_id, self.sensor_id, self.band_name)


class Fit_Association(Base):
    """Associates a collection of Fit objects
    with a particular analysis.
    
    """
    __tablename__ = "Fit_Association"

    @classmethod
    def creator(cls, discriminator):
        """Provide a 'creator' function to use with 
        the association proxy."""

        return lambda fits:Fit_Association(
                                fits=fits, 
                                discriminator=discriminator)

    discriminator = Column(String(100))
    """Refers to the type of analysis."""

    @property
    def analysis(self):
        """Return the analysis object."""
        return getattr(self, "%s_analysis" % self.discriminator)

class HasFits(object):
    """HasFits mixin, creates a relationship to
    the address_association table for each parent.
    
    """
    @declared_attr
    def fit_association_id(cls):
        return Column(Integer, ForeignKey("Fit_Association.id"))

    @declared_attr
    def fit_association(cls):
        discriminator = cls.__name__.lower()
        cls.fits= association_proxy(
                    "fit_association", "fits",
                    creator=Fit_Association.creator(discriminator)
                )
        return relationship(Fit_Association, 
                    backref=backref("%s_analysis" % discriminator, 
                                        uselist=False))

class Avalanche(HasFits,Base):
    spatial_sample = Column(String(100))
    temporal_sample = Column(String(100))
    threshold_mode = Column(String(100))
    threshold_level = Column(Float)
    time_scale = Column(Float)
    event_method = Column(String(100))
    cascade_method = Column(String(100))

    n = Column(Integer)

    interevent_intervals_mean = Column(Float)
    interevent_intervals_median = Column(Float)
    interevent_intervals_mode = Column(Float)

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
    
    subject_id = Column(Integer, ForeignKey('Subject.id'))
    task_id = Column(Integer, ForeignKey('Task.id'))
    experiment_id = Column(Integer, ForeignKey('Experiment.id'))
    recording_id = Column(Integer, ForeignKey('Recording.id'))
    filter_id = Column(Integer, ForeignKey('Filter.id'))
    sensor_id = Column(Integer, ForeignKey('Sensor.id'))

    subject = relationship(Subject, backref=backref('avalanches')) #, order_by=id))
    task = relationship(Task, backref=backref('avalanches')) #, order_by=id))
    experiment = relationship(Experiment, backref=backref('avalanches')) #, order_by=id))
    sensor = relationship(Sensor, backref=backref('avalanches')) #, order_by=id))
    recording = relationship(Recording, backref=backref('avalanches')) #, order_by=id))
    filter = relationship(Filter, backref=backref('avalanches')) #, order_by=id))

    def __repr__(self):
        return "<%s(subject='%s', experiment='%s', task='%s', sensor='%s', threshold='%s', timescale='%s')>" % \
                (self.__class__.__name__, self.subject_id, self.experiment_id, self.task_id, self.sensor_id, self.threshold_level, self.time_scale)

class Fit(Base):
    analysis_type = Column(String(100)) 
    variable = Column(String(100)) 
    distribution = Column(String(100)) 
    parameter1_name = Column(String(100)) 
    parameter1_value = Column(Float) 
    parameter2_name = Column(String(100)) 
    parameter2_value = Column(Float) 
    parameter3_name = Column(String(100))
    parameter3_value = Column(Float)
    xmin = Column(Float) 
    xmax = Column(Float)
    loglikelihood = Column(Float) 
    loglikelihood_ratio = Column(Float) 
    KS = Column(Float)
    D_plus_critical_branching = Column(Float)
    D_minus_critical_branching = Column(Float)
    Kappa = Column(Float)
    p = Column(Float)
    n_tail = Column(Integer)
    noise_flag = Column(Boolean)
    discrete = Column(Boolean)
    
    analysis_id = Column(Integer)
    subject_id = Column(Integer, ForeignKey('Subject.id'))
    task_id = Column(Integer, ForeignKey('Task.id'))
    experiment_id = Column(Integer, ForeignKey('Experiment.id'))
    recording_id = Column(Integer, ForeignKey('Recording.id'))
    filter_id = Column(Integer, ForeignKey('Filter.id'))
    sensor_id = Column(Integer, ForeignKey('Sensor.id'))

    association_id = Column(Integer, ForeignKey("Fit_Association.id"))
    association = relationship(Fit_Association, backref="fits")
    analysis = association_proxy("association", "analysis")

    subject = relationship(Subject, backref=backref('fits')) #, order_by=id))
    task = relationship(Task, backref=backref('fits')) #, order_by=id))
    experiment = relationship(Experiment, backref=backref('fits')) #, order_by=id))
    sensor = relationship(Sensor, backref=backref('fits')) #, order_by=id))
    recording = relationship(Recording, backref=backref('fits')) #, order_by=id))
    filter = relationship(Filter, backref=backref('fits')) #, order_by=id))

    def __repr__(self):
        return "<%s(subject='%s', experiment='%s', task='%s', sensor='%s', variable='%s', distribution='%s')>" % \
                (self.__class__.__name__, self.subject_id, self.experiment_id, self.task_id, self.sensor_id, self.variable, self.distribution)
