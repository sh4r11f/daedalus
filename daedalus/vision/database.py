# -*- coding: utf-8 -*-
# ======================================================================================== #
#
#
#                    SCRIPT: database.py
#
#
#          DESCRIPTION: Classes for SQL database management
#
#
#                       RULE: DAYW
#
#
#
#                  CREATOR: Sharif Saleki
#                         TIME: 06-07-2024-[78 105 98 105114117]
#                       SPACE: Dartmouth College, Hanover, NH
#
# ======================================================================================== #
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, ForeignKey, CheckConstraint, UniqueConstraint, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# Base class for our ORM models
Base = declarative_base()


class Author(Base):
    """
    This class represents an Author in the database.

    Attributes:
        id (int): The primary key.
        name (str): The name of the author.
        email (str): The email of the author.
        affiliation (str): The affiliation of the author.
        location (str): The location of the author.
    """
    __tablename__ = 'authors'
    id = Column(Integer, primary_key=True)
    email = Column(String, nullable=False, unique=True)
    name = Column(String)
    affiliation = Column(String)
    location = Column(String)


class ExperimentAuthor(Base):
    """
    This class represents the many-to-many relationship between Experiments and Authors.

    Attributes:
        experiment_id (int): The ID of the experiment.
        author_id (int): The ID of the author.
    """
    __tablename__ = 'experiment_authors'

    experiment_id = Column(Integer, ForeignKey('experiments.id'), primary_key=True)
    author_id = Column(Integer, ForeignKey('authors.id'), primary_key=True)


class Experiment(Base):
    """
    This class represents an Experiment in the database.

    Attributes:
        id (int): The primary key.
        title (str): The title of the experiment.
        shorthand (str): The shorthand name of the experiment.
        repository (str): The repository link of the experiment.
        experiment_type (str): The type of the experiment.
        version (str): The version of the experiment.
        description (str): The description of the experiment.
        n_subjects (int): The number of subjects in the experiment.
        n_sessions (int): The number of sessions in the experiment.
    """
    __tablename__ = 'experiments'

    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    repository = Column(String)
    shorthand = Column(String)
    experiment_type = Column(String)
    version = Column(String)
    description = Column(String)
    n_subjects = Column(Integer)
    n_sessions = Column(Integer)

    # Relationships
    authors = relationship('Author', secondary='experiment_authors', back_populates='experiments')
    subjects = relationship('Subject', back_populates='experiments')
    directories = relationship('Directory', back_populates='experiment')

    __table_args__ = (
        CheckConstraint("n_subjects > 0", name="n_subjects_check"),
        CheckConstraint("n_sessions > 0", name="n_sessions_check"),
        CheckConstraint("experiment_type in ('Behavioral', 'EyeTracking', 'Electrophysiology')", name="type_check"),
        UniqueConstraint('title', 'repository', name='unique_experiment')
    )

    def __repr__(self):
        return f'<Experiment(id={self.id}, title={self.title}, shorthand={self.shorthand}, repository={self.repository})>'


class Directory(Base):
    """
    This class represents a Directory in the database.

    Attributes:
        id (int): The primary key.
        name (str): The name of the directory.
        path (str): The path of the directory.
    """
    __tablename__ = 'directories'

    id = Column(Integer, primary_key=True)
    path = Column(String, nullable=False, unique=True)
    name = Column(String)

    # Parents
    experiment_id = Column(Integer, ForeignKey('experiments.id'), nullable=False)
    experiment = relationship('Experiment', back_populates='directories')

    # Children
    files = relationship('File', back_populates='directory')

    def __repr__(self):
        return f'<Directory(id={self.id}, name={self.name}, path={self.path})>'


class File(Base):
    """
    This class represents a File in the database.

    Attributes:
        id (int): The primary key.
        name (str): The name of the file.
        path (str): The path of the file.
        size (int): The size of the file in bytes.
    """
    __tablename__ = 'files'

    id = Column(Integer, primary_key=True)
    path = Column(String, nullable=False, unique=True)
    name = Column(String)

    # Parents
    directory_id = Column(Integer, ForeignKey('directories.id'), nullable=False)
    directory = relationship('Directory', back_populates='files')

    def __repr__(self):
        return f'<File(id={self.id}, name={self.name}, path={self.path}, size={self.size})>'


class Subject(Base):
    """
    This class represents a Subject in the database.

    Attributes:
        id (int): The primary key.
        initials (str): The initials of the subject.
        name (str): The name of the subject.
        age (int): The age of the subject.
        gender (str): The gender of the subject.
        netid (str): The network ID of the subject.
        email (str): The email of the subject.
        vision (str): The vision status of the subject.
        dominant_eye (str): The dominant eye of the subject.
        dominant_hand (str): The dominant hand of the subject.
    """
    __tablename__ = 'subjects'

    id = Column(Integer, primary_key=True)
    initials = Column(String, nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String, nullable=False)
    vision = Column(String, nullable=False)
    name = Column(String)
    netid = Column(String)
    email = Column(String)
    dominant_eye = Column(String)
    dominant_hand = Column(String)
    registered_at = Column(DateTime, default=datetime.now)

    # Parents
    experiment_id = Column(Integer, ForeignKey('experiments.id'), nullable=False)
    experiments = relationship('Experiment', back_populates='subjects')

    # Children
    tasks = relationship('Task', secondary='subject_tasks', back_populates='subjects')

    __table_args__ = (
        CheckConstraint("age > 0", name="age_check"),
        CheckConstraint("gender in ('Male', 'Female', 'NB')", name="gender_check"),
        CheckConstraint("vision in ('Normal', 'Corrected', 'Impaired')", name="vision_check"),
        CheckConstraint("dominant_eye in ('Right', 'Left')", name="dominant_eye_check"),
        CheckConstraint("dominant_hand in ('Right', 'Left', 'Both')", name="dominant_hand_check"),
        UniqueConstraint('initials', 'age', 'gender', name='unique_subject')
    )

    def __repr__(self):
        return f'<Subject(id={self.id}, initials={self.initials}, name={self.name}, age={self.age})>'


class Task(Base):
    """
    This class represents a Task in the database.

    Attributes:
        id (int): The primary key.
        name (str): The name of the task.
        description (str): The description of the task.
        duration (int): The duration of the task in minutes.
        parameters (str): The parameters of the task.
    """
    __tablename__ = 'tasks'

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    parameters = Column(String, nullable=False)
    description = Column(String)
    created_at = Column(DateTime, default=datetime.now)

    # Parents
    subject_id = Column(Integer, ForeignKey('subjects.id'), nullable=False)
    subject = relationship('Subject', back_populates='tasks')

    # Children
    stages = relationship('Stage', back_populates='task')

    __table_args__ = (
        CheckConstraint("duration > 0", name="duration_check")
    )

    def __repr__(self):
        return f'<Task(id={self.id}, name={self.name}, description={self.description}, duration={self.duration})>'


class Stage(Base):
    """
    This class represents a Stage in the database.

    Attributes:
        id (int): The primary key.
        name (str): The name of the stage.
        order (int): The order of the stage.
        description (str): The description of the stage.
    """
    __tablename__ = 'stages'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    order = Column(Integer, nullable=False)
    description = Column(String)

    # Relationships
    task_id = Column(Integer, ForeignKey('tasks.id'), nullable=False)
    task = relationship('Task', back_populates='stages')
    blocks = relationship('Block', back_populates='stage')

    __table_args__ = (
        UniqueConstraint('order', "task_id", name='unique_stage')
    )

    def __repr__(self):
        return f'<Stage(id={self.id}, name={self.name}, description={self.description}, order={self.order})>'


class Block(Base):
    """
    This class represents a Block in the database.

    Attributes:
        id (int): The primary key.
        order (int): The order of the block.
    """
    __tablename__ = 'blocks'

    id = Column(Integer, primary_key=True)
    order = Column(Integer, nullable=False)

    # Relationships
    stage_id = Column(Integer, ForeignKey('stages.id'), nullable=False)
    stage = relationship('Stage', back_populates='blocks')
    trials = relationship('Trial', back_populates='block')

    __table_args__ = (
        UniqueConstraint('order', 'stage_id', name='unique_block')
    )

    def __repr__(self):
        return f'<Block(id={self.id}, order={self.order})>'


class Trial(Base):
    """
    This class represents a Trial in the database.

    Attributes:
        id (int): The primary key.
        order (int): The order of the trial.
    """
    __tablename__ = 'trials'

    id = Column(Integer, primary_key=True)
    order = Column(Integer, nullable=False)

    # Relationships
    block_id = Column(Integer, ForeignKey('blocks.id'), nullable=False)
    block = relationship('Block', back_populates='trials')
    stimuli = relationship('Stimulus', back_populates='trial')
    behavioral_responses = relationship('BehavioralResponse', back_populates='trial')
    eyetracking_events = relationship('EyeTrackingEvent', back_populates='trial')
    eyetracking_samples = relationship('EyeTrackingSample', back_populates='trial')
    ephys_samples = relationship('EphysSample', back_populates='trial')

    __table_args__ = (
        UniqueConstraint('order', 'block_id', name='unique_trial')
    )

    def __repr__(self):
        return f'<Trial(id={self.id}, order={self.order})>'


class Stimulus(Base):
    """
    This class represents a Stimulus in the database.

    Attributes:
        id (int): The primary key.
        name (str): The name of the stimulus.
    """
    __tablename__ = 'stimuli'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)

    # Relationships
    trial_id = Column(Integer, ForeignKey('trials.id'), nullable=False)
    trial = relationship('Trial', back_populates='stimuli')

    

    def __repr__(self):
        return f'<Stimulus(id={self.id}, name={self.name})>'


class EyeTrackingEvent(Base):
    """
    This class represents an Eye-Tracking Event in the database.

    Attributes:
        id (int): The primary key.
        event_type (str): The type of the event.
        time_start (float): The start time of the event.
        time_end (float): The end time of the event.
        duration (float): The duration of the event.
        gaze_start_x (float): The starting x-coordinate of the gaze.
        gaze_start_y (float): The starting y-coordinate of the gaze.
        gaze_end_x (float): The ending x-coordinate of the gaze.
        gaze_end_y (float): The ending y-coordinate of the gaze.
        gaze_avg_x (float): The average x-coordinate of the gaze.
        gaze_avg_y (float): The average y-coordinate of the gaze.
        ppd_start_x (float): The starting x-coordinate of the pupil position.
        ppd_start_y (float): The starting y-coordinate of the pupil position.
        ppd_end_x (float): The ending x-coordinate of the pupil position.
        ppd_end_y (float): The ending y-coordinate of the pupil position.
        ppd_avg_x (float): The average x-coordinate of the pupil position.
        ppd_avg_y (float): The average y-coordinate of the pupil position.
        deg_avg_x (float): The average x-coordinate in degrees.
        deg_avg_y (float): The average y-coordinate in degrees.
        pupil_start (float): The starting pupil size.
        pupil_end (float): The ending pupil size.
        pupil_avg (float): The average pupil size.
        saccade_amplitude (float): The amplitude of the saccade.
        saccade_angle (float): The angle of the saccade.
        saccade_velocity_start (float): The starting velocity of the saccade.
        saccade_velocity_end (float): The ending velocity of the saccade.
        saccade_velocity_avg (float): The average velocity of the saccade.
        saccade_velocity_peak (float): The peak velocity of the saccade.
    """
    __tablename__ = 'eyetracking_events'

    id = Column(Integer, primary_key=True)
    event_type = Column(String, nullable=False)
    time_start = Column(Float)
    time_end = Column(Float)
    duration = Column(Float)
    gaze_start_x = Column(Float)
    gaze_start_y = Column(Float)
    gaze_end_x = Column(Float)
    gaze_end_y = Column(Float)
    gaze_avg_x = Column(Float)
    gaze_avg_y = Column(Float)
    ppd_start_x = Column(Float)
    ppd_start_y = Column(Float)
    ppd_end_x = Column(Float)
    ppd_end_y = Column(Float)
    ppd_avg_x = Column(Float)
    ppd_avg_y = Column(Float)
    deg_avg_x = Column(Float)
    deg_avg_y = Column(Float)
    pupil_start = Column(Float)
    pupil_end = Column(Float)
    pupil_avg = Column(Float)
    saccade_amplitude = Column(Float)
    saccade_angle = Column(Float)
    saccade_velocity_start = Column(Float)
    saccade_velocity_end = Column(Float)
    saccade_velocity_avg = Column(Float)
    saccade_velocity_peak = Column(Float)

    # Relationships
    trial_id = Column(Integer, ForeignKey('trials.id'))
    trial = relationship('Trial', back_populates='eyetracking_events')
    samples = relationship('EyeTrackingSample', back_populates='event')

    def __repr__(self):
        return f'<EyeTrackingEvent(id={self.id}, event_type={self.event_type})>'


class EyeTrackingSample(Base):
    """
    This class represents an Eye-Tracking Sample in the database.

    Attributes:
        id (int): The primary key.
        timestamp (float): The timestamp of the sample.
        gaze_x (float): The x-coordinate of the gaze.
        gaze_y (float): The y-coordinate of the gaze.
        ppd_x (float): The x-coordinate of the pupil position.
        ppd_y (float): The y-coordinate of the pupil position.
        pupil (float): The size of the pupil.
    """
    __tablename__ = 'eyetracking_samples'

    id = Column(Integer, primary_key=True)
    timestamp = Column(Float)
    gaze_x = Column(Float)
    gaze_y = Column(Float)
    ppd_x = Column(Float)
    ppd_y = Column(Float)
    pupil = Column(Float)

    # Relationships
    trial_id = Column(Integer, ForeignKey('trials.id'))
    trial = relationship('Trial', back_populates='eyetracking_samples')
    event_id = Column(Integer, ForeignKey('eyetracking_events.id'))
    event = relationship('EyeTrackingEvent', back_populates='samples')

    def __repr__(self):
        return f'<EyeTrackingSample(id={self.id}, timestamp={self.timestamp})>'


class BehavioralResponse(Base):
    """
    This class represents a Behavioral Response in the database.

    Attributes:
        id (int): The primary key.
        response (str): The response of the subject.
    """
    __tablename__ = 'behavioral_responses'

    id = Column(Integer, primary_key=True)
    choice = Column(String)

    # Relationships
    trial_id = Column(Integer, ForeignKey('trials.id'))
    trial = relationship('Trial', back_populates='behavioral_responses')

    def __repr__(self):
        return f'<BehavioralResponse(id={self.id}, response={self.response})>'


class EphysSample(Base):
    """
    This class represents an Electrophysiological Sample in the database.

    Attributes:
        id (int): The primary key.
        timestamp (float): The timestamp of the sample.
        channel (str): The channel of the sample.
        value (float): The value of the sample.
    """
    __tablename__ = 'ephys_samples'

    id = Column(Integer, primary_key=True)
    timestamp = Column(Float)
    channel = Column(String)
    value = Column(Float)

    # Relationships
    trial_id = Column(Integer, ForeignKey('trials.id'))
    trial = relationship('Trial', back_populates='ephys_samples')

    def __repr__(self):
        return f'<EphysSample(id={self.id}, timestamp={self.timestamp}, channel={self.channel}, value={self.value})>'
