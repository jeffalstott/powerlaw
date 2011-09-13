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
        return cls.__name__+'s'
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

    subject_id = Column(Integer, ForeignKey('Subjects.id'))
    task_id = Column(Integer, ForeignKey('Tasks.id'))

    subject = relationship(Subject, backref=backref('experiments', order_by=id))
    task = relationship(Task, backref=backref('experiments', order_by=id))

    def __repr__(self):
        return "<%s(subject='%s', visit='%s', task='%s')>" % \
                (self.__class__.__name__, self.subject_id, self.visit_number, self.task_id)

class Recording(Base):
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

    def __repr__(self):
        return "<%s(subject='%s', experiment='%s', task='%s', sensor='%s')>" % \
                (self.__class__.__name__, self.subject_id, self.experiment_id, self.task_id, self.sensor_id)

class Filter(Base):
    filter_type = Column(String(100))
    poles = Column(Integer)
    window = Column(Integer)
    band_name = Column(String(100))
    band_min = Column(Float)
    band_max = Column(Float)
    duration = Column(Float)
    notch = Column(Boolean)
    phase_shuffled = Column(Boolean)

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

    def __repr__(self):
        return "<%s(subject='%s', experiment='%s', task='%s', sensor='%s', band='%s')>" % \
                (self.__class__.__name__, self.subject_id, self.experiment_id, self.task_id, self.sensor_id, self.band_name)


class Fit_Association(Base):
    """Associates a collection of Fit objects
    with a particular analysis.
    
    """
    __tablename__ = "Fit_Associations"

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
        return Column(Integer, ForeignKey("fit_association.id"))

    @declared_attr
    def fit_association(cls):
        discriminator = cls.__name__.lower()
        cls.fits= association_proxy(
                    "fit_association", "fit",
                    creator=Fit_Association.creator(discriminator)
                )
        return relationship(Fit_Association, 
                    backref=backref("%s_analysis" % discriminator, 
                                        uselist=False))

class Avalanche(HasFits,Base):
    subsample = Column(String(100))
    threshold_mode = Column(String(100))
    threshold_level = Column(Float)
    time_scale = Column(Float)
    event_method = Column(String(100))
    cascade_method = Column(String(100))
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

    subject = relationship(Subject, backref=backref('avalanches', order_by=id))
    task = relationship(Task, backref=backref('avalanches', order_by=id))
    experiment = relationship(Experiment, backref=backref('avalanches', order_by=id))
    sensor = relationship(Sensor, backref=backref('avalanches', order_by=id))
    recording = relationship(Recording, backref=backref('avalanches', order_by=id))
    filter = relationship(Filter, backref=backref('avalanches', order_by=id))

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
    KS = Column(Float)
    p = Column(Float)
    
    subject_id = Column(Integer, ForeignKey('Subjects.id'))
    task_id = Column(Integer, ForeignKey('Tasks.id'))
    experiment_id = Column(Integer, ForeignKey('Experiments.id'))
    recording_id = Column(Integer, ForeignKey('Recordings.id'))
    filter_id = Column(Integer, ForeignKey('Filters.id'))
    sensor_id = Column(Integer, ForeignKey('Sensors.id'))

    association_id = Column(Integer, ForeignKey("Fit_Association.id"))
    association = relationship(Fit_Association, backref="fits")
    analysis = association_proxy("association", "analysis")

    subject = relationship(Subject, backref=backref('fits', order_by=id))
    task = relationship(Task, backref=backref('fits', order_by=id))
    experiment = relationship(Experiment, backref=backref('fits', order_by=id))
    sensor = relationship(Sensor, backref=backref('fits', order_by=id))
    recording = relationship(Recording, backref=backref('fits', order_by=id))
    filter = relationship(Filter, backref=backref('fits', order_by=id))

    def __repr__(self):
        return "<%s(subject='%s', experiment='%s', task='%s', sensor='%s', variable='%s', distribution='%s')>" % \
                (self.__class__.__name__, self.subject_id, self.experiment_id, self.task_id, self.sensor_id, self.variable, self.distribution)
