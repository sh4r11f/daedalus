# -*- coding: utf-8 -*-
# ======================================================================================== #
#
#
#                    SCRIPT: experiments.py
#
#
#          DESCRIPTION: Database of experiments and their parameters
#
#
#                       RULE: DAYW
#
#
#
#                  CREATOR: Sharif Saleki
#                         TIME: 06-10-2024-[78 105 98 105114117]
#                       SPACE: Dartmouth College, Hanover, NH
#
# ======================================================================================== #
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from daedalus.data.database import (
    Base, Experiment, Author, ExperimentAuthor, Directory, File,
    Subject, SubjectSession, Task, TaskParameter,
    Block, Trial, Stimulus, StimulusProperty,
    BehavioralResponse
)


class ExperimentDatabase:
    """
    Base class for handling the psychophysics database operations.

    Attributes:
        db_path (str): The path to the SQLite database file.
        engine: The SQLAlchemy engine instance.
        Session: The SQLAlchemy session maker.
    """
    def __init__(self, db_path, exp_id):
        """
        Initialize the database connection and create tables.

        Args:
            db_path (str): Path to the SQLite database file.
        """
        db_url = f"sqlite:///{db_path}"

        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        self.exp_type = "Psychophysics"
        self.exp_id = exp_id

    def add_experiment(self, title, shorthand, repository, version, description, n_subjects, n_sessions):
        """
        Add a new experiment to the database.

        Args:
            title (str): The title of the experiment.
            shorthand (str): The shorthand name of the experiment.
            repository (str): The repository link of the experiment.
            experiment_type (str): The type of the experiment.
            version (str): The version of the experiment.
            description (str): The description of the experiment.
            n_subjects (int): The number of subjects in the experiment.
            n_sessions (int): The number of sessions in the experiment.

        Returns:
            int: The ID of the added experiment.
        """
        session = self.Session()
        experiment = Experiment(
            experiment_type=self.exp_type,
            id=self.exp_id,
            title=title,
            shorthand=shorthand,
            repository=repository,
            version=version,
            description=description,
            n_subjects=n_subjects,
            n_sessions=n_sessions
        )
        try:
            session.add(experiment)
            session.commit()
        except IntegrityError:
            session.rollback()
            existing_exp = session.query(Experiment).filter_by(title=title, repository=repository).first()
            return existing_exp.id
        return experiment.id

    def add_author(self, name, email, affiliation, location):
        """
        Add a new author to the database.

        Args:
            name (str): The name of the author.
            email (str): The email of the author.
            affiliation (str): The affiliation of the author.
            location (str): The location of the author.

        Returns:
            int: The ID of the added author.
        """
        session = self.Session()
        author = Author(name=name, email=email, affiliation=affiliation, location=location)
        try:
            session.add(author)
            session.commit()
        except IntegrityError:
            existing_author = session.query(Author).filter_by(email=email).first()
            return existing_author.id
        return author.id

    def add_experiment_author(self, experiment_id, author_id):
        """
        Add an author to an experiment (many-to-many relationship).

        Args:
            experiment_id (int): The ID of the experiment.
            author_id (int): The ID of the author.

        Returns:
            None
        """
        session = self.Session()
        experiment_author = ExperimentAuthor(experiment_id=experiment_id, author_id=author_id)
        session.add(experiment_author)
        session.commit()

    def add_directory(self, name, path, experiment_id):
        """
        Add a new directory to the database.

        Args:
            name (str): The name of the directory.
            path (str): The path of the directory.
            experiment_id (int): The ID of the experiment.

        Returns:
            int: The ID of the added directory.
        """
        session = self.Session()
        directory = Directory(name=name, path=path, experiment_id=experiment_id)
        try:
            session.add(directory)
            session.commit()
        except IntegrityError:
            existing_dir = session.query(Directory).filter_by(path=path).first()
            return existing_dir.id
        return directory.id

    def add_file(self, name, path, directory_id):
        """
        Add a new file to the database.

        Args:
            name (str): The name of the file.
            path (str): The path of the file.
            directory_id (int): The ID of the directory.

        Returns:
            int: The ID of the added file.
        """
        session = self.Session()
        file = File(name=name, path=path, directory_id=directory_id)
        try:
            session.add(file)
            session.commit()
        except IntegrityError:
            existing_file = session.query(File).filter_by(path=path).first()
            return existing_file.id
        return file.id

    def add_subject(self, initials, age, gender, vision, **kwargs):
        """
        Add a new subject to the database.

        Args:
            initials (str): The initials of the subject.
            age (int): The age of the subject.
            gender (str): The gender of the subject.
            vision (str): The vision status of the subject.
            name (str, optional): The name of the subject.
            netid (str, optional): The network ID of the subject.
            email (str, optional): The email of the subject.
            dominant_eye (str, optional): The dominant eye of the subject.
            dominant_hand (str, optional): The dominant hand of the subject.

        Returns:
            int: The ID of the added subject.
        """
        session = self.Session()
        subject = Subject(
            initials=initials,
            age=age,
            gender=gender,
            vision=vision,
            **kwargs
        )
        try:
            session.add(subject)
            session.commit()
        except IntegrityError:
            existing_subject = session.query(Subject).filter_by(initials=initials, age=age, gender=gender).first()
            return existing_subject.id
        return subject.id

    def add_subject_session(self, subject_id, session_num):
        """
        Add a new subject session to the database.

        Args:
            subject_id (int): The ID of the subject.
            session_num (int): The number of the session.

        Returns:
            int: The ID of the added subject session.
        """
        session = self.Session()
        subject_session = SubjectSession(subject_id=subject_id, session_num=session_num)
        try:
            session.add(subject_session)
            session.commit()
        except IntegrityError:
            existing_subject_session = session.query(SubjectSession).filter_by(
                subject_id=subject_id, session=session_num).first()
            return existing_subject_session.id
        return subject_session.id

    def add_task(self, name, **kwargs):
        """
        Add a new task to the database.

        Args:
            name (str): The name of the task.

        Returns:
            int: The ID of the added task.
        """
        session = self.Session()
        task = Task(name=name, **kwargs)
        try:
            session.add(task)
            session.commit()
        except IntegrityError:
            existing_task = session.query(Task).filter_by(name=name).first()
            return existing_task.id
        return task.id

    def add_task_parameter(self, task_id, key, value):
        """
        Add a new parameter to a task.

        Args:
            task_id (int): The ID of the task.
            key (str): The parameter key.
            value (str): The parameter value.
        """
        session = self.Session()
        task_parameter = TaskParameter(task_id=task_id, key=key, value=value)
        try:
            session.add(task_parameter)
            session.commit()
        except IntegrityError:
            existing_task_parameter = session.query(TaskParameter).filter_by(task_id=task_id, key=key).first()
            return existing_task_parameter.id
        return task_parameter.id

    def add_block(self, order, stage_id):
        """
        Add a new block to the database.

        Args:
            order (int): The order of the block.
            stage_id (int): The ID of the stage.

        Returns:
            int: The ID of the added block.
        """
        session = self.Session()
        block = Block(order=order, stage_id=stage_id)
        try:
            session.add(block)
            session.commit()
        except IntegrityError:
            existing_block = session.query(Block).filter_by(order=order, stage_id=stage_id).first()
            return existing_block.id
        return block.id

    def add_trial(self, order, block_id):
        """
        Add a new trial to the database.

        Args:
            order (int): The order of the trial.
            block_id (int): The ID of the block.

        Returns:
            int: The ID of the added trial.
        """
        session = self.Session()
        trial = Trial(order=order, block_id=block_id)
        try:
            session.add(trial)
            session.commit()
        except IntegrityError:
            existing_trial = session.query(Trial).filter_by(order=order, block_id=block_id).first()
            return existing_trial.id
        return trial.id

    def add_stimulus(self, name, trial_id):
        """
        Add a new stimulus to the database.

        Args:
            name (str): The name of the stimulus.
            trial_id (int): The ID of the trial.

        Returns:
            int: The ID of the added stimulus.
        """
        session = self.Session()
        stimulus = Stimulus(name=name, trial_id=trial_id)
        try:
            session.add(stimulus)
            session.commit()
        except IntegrityError:
            existing_stimulus = session.query(Stimulus).filter_by(name=name, trial_id=trial_id).first()
            return existing_stimulus.id
        return stimulus.id

    def add_stimulus_property(self, stim_id, key, value):
        """
        Add a new property to a stimulus.

        Args:
            stim_id (int): The ID of the stimulus.
            key (str): The property key.
            value (str): The property value.
        """
        session = self.Session()
        stimulus_property = StimulusProperty(stimulus_id=stim_id, key=key, value=value)
        try:
            session.add(stimulus_property)
            session.commit()
        except IntegrityError:
            existing_stimulus_property = session.query(StimulusProperty).filter_by(stimulus_id=stim_id, key=key).first()
            return existing_stimulus_property.id
        return stimulus_property.id

    def add_behavioral_response(self, response, trial_id):
        """
        Add a new behavioral response to the database.

        Args:
            response (str): The behavioral response.
            trial_id (int): The ID of the trial.

        Returns:
            int: The ID of the added behavioral response.
        """
        session = self.Session()
        behavioral_response = BehavioralResponse(response=response, trial_id=trial_id)
        session.add(behavioral_response)
        session.commit()
        return behavioral_response.id

    def get_experiment(self):
        """
        Retrieve all experiments from the database.

        Returns:
            list: List of experiments.
        """
        session = self.Session()
        return session.query(Experiment).first()

    def get_subjects(self):
        """
        Retrieve all subjects from the database.

        Returns:
            list: List of subjects.
        """
        session = self.Session()
        return session.query(Subject).all()

    def get_tasks(self, experiment_id):
        """
        Retrieve all tasks for a given experiment.

        Args:
            experiment_id (int): The ID of the experiment.

        Returns:
            list: List of tasks.
        """
        session = self.Session()
        return session.query(Task).filter_by(experiment_id=experiment_id).all()

    def get_blocks(self, stage_id):
        """
        Retrieve all blocks for a given stage.

        Args:
            stage_id (int): The ID of the stage.

        Returns:
            list: List of blocks.
        """
        session = self.Session()
        return session.query(Block).filter_by(stage_id=stage_id).all()

    def get_trials(self, block_id):
        """
        Retrieve all trials for a given block.

        Args:
            block_id (int): The ID of the block.

        Returns:
            list: List of trials.
        """
        session = self.Session()
        return session.query(Trial).filter_by(block_id=block_id).all()

    def get_stimuli(self, trial_id):
        """
        Retrieve all stimuli for a given trial.

        Args:
            trial_id (int): The ID of the trial.

        Returns:
            list: List of stimuli.
        """
        session = self.Session()
        return session.query(Stimulus).filter_by(trial_id=trial_id).all()

    def get_stimulus_properties(self, stim_id):
        """
        Retrieve all properties for a given stimulus.

        Args:
            stim_id (int): The ID of the stimulus.

        Returns:
            list: List of stimulus properties.
        """
        session = self.Session()
        return session.query(StimulusProperty).filter_by(stimulus_id=stim_id).all()

    def get_behavioral_responses(self, trial_id):
        """
        Retrieve all behavioral responses for a given trial.

        Args:
            trial_id (int): The ID of the trial.

        Returns:
            list: List of behavioral responses.
        """
        session = self.Session()
        return session.query(BehavioralResponse).filter_by(trial_id=trial_id).all()

    @staticmethod
    def get_table_fields(model_class):
        """
        Get all fields (column names) in a table.

        Args:
            model_class (Base): The SQLAlchemy model class for the table.

        Returns:
            list: List of column names.
        """
        return model_class.__table__.columns.keys()

    def export_subject_data(self, subject_id, file_path):
        """
        Export data for a specific block to a CSV or TSV file.

        Args:
            block_id (int): The ID of the block.
            file_path (str): The path to the exported file.

        Returns:
            str: The path to the exported file.
        """
        session = self.Session()
        exp = session.query(Experiment).filter_by(id=self.experiment_id).first()
        subject = session.query(Subject).filter_by(id=subject_id).first()
        tasks = session.query(Task).filter_by(subject_id=subject.id).all()

        if not tasks:
            return None

        # Collect data
        data = []
        for task in tasks:
            blocks = session.query(Block).filter_by(stage_id=task.id).all()
            for block in blocks:
                trials = session.query(Trial).filter_by(block_id=block.id).all()
                for trial in trials:
                    stimuli = session.query(Stimulus).filter_by(trial_id=trial.id).all()
                    for stimulus in stimuli:

                        # Information
                        trial_data = dict(
                            ExperimentName=exp.title,
                            ExperimentVersion=exp.version,
                            SubjectID=subject.id,
                            SubjectInitials=subject.initials,
                            TaskName=task.name,
                            TaskDate=task.date,
                            BlockNumber=block.order,
                            TrialNumber=trial.order,
                            StimulusName=stimulus.name
                            )

                        # Stimulus
                        stimulus_fields = self.get_table_fields(Stimulus)
                        for field in stimulus_fields:
                            if field not in ['id', 'trial_id', 'trial']:
                                trial_data[field] = getattr(stimulus, field)

                        # Response
                        behavioral_response = session.query(BehavioralResponse).filter_by(trial_id=trial.id).first()
                        trial_data['Response'] = behavioral_response.choice

                        # Append to data
                        data.append(trial_data)

        # Create DataFrame
        df = pd.DataFrame(data)

        # Export to file
        filename = Path(file_path) / f'subject-{subject_id:02d}_exp-{self.exp_id}.csv'
        df.to_csv(filename, sep=',', index=False)

        return filename

    def close(self):
        """
        Close the database connection properly and save the database.
        """
        self.Session.close_all()
        self.engine.dispose()
