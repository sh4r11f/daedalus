# -*- coding: utf-8 -*-
# ==================================================================================================== #
#
#
#                    SCRIPT: psyphy.py
#
#
#          DESCRIPTION: Class for psychophysics experiments
#
#
#                       RULE: DAYW
#
#
#
#                  CREATOR: Sharif Saleki
#                         TIME: 05-26-2024-[78 105 98 105114117]
#                       SPACE: Dartmouth College, Hanover, NH
#
# ==================================================================================================== #
import sys
from pathlib import Path
from typing import Dict, Union, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from psychopy import core, monitors, visual, event, info, gui, data
from psychopy.iohub.devices import Computer

import logging
from daedalus import log_handler, utils

from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from database import (
    Base, Experiment, Author, ExperimentAuthor, Directory, File,
    Subject, SubjectSession, Task, TaskParameter,
    Block, Trial, Stimulus, StimulusProperty,
    BehavioralResponse
)


class PsychoPhysicsExperiment:
    """
    Template experiment with basic functionality.

    Mostly deals with setting up primitive objects and settings like subject info, system checks, files and paths,
    logging, etc.

    Every experiment needs the following attributes/functionalities:
        - Directories and files.
        - Settings that are specific to the experiment (metadata, conditions, etc.).
            An example of this can be found in docs/examples. Settings and parameters file should live under
            config/ directory of the experiment.
        - Getting user input.
        - A monitor and a window object (from psychopy).

    Args:
        root (str or Path): Home directory of the experiment.
        verison (float): Experiment version
        debug (bool): Debug mode for the experiment.
        info (dict): Subject and experiment information.
    """
    def __init__(self, name: str, root: Union[str, Path], version: str, platform: str, debug: bool):

        # Setup
        self.exp_type = 'psychophysics'
        self.name = name
        self.root = Path(root)
        self.version = version
        self.platform = platform
        self.debug = debug

        # Directories and files
        self.dirs = self.init_directories()
        self.files = self.init_files()

        # Load config
        self.settings = utils.read_config(self.files["settings"])
        # easier access to the platform settings
        self.platform_settings = self.settings["Platforms"][self.platform]
        self.exp_params = utils.read_config(self.files["experiment_params"])
        self.stim_params = utils.read_config(self.files["stimuli_params"])
        monitor_name = self.platfrom_settings["Monitor"]
        mon_params = utils.read_config(self.files["monitors"])
        self.monitor_params = mon_params[monitor_name]

        # Database
        self.db = None
        self.logger = None
        self.monitor = None
        self.window = None
        self.clocks = None

    def init_database(self):
        """
        Initialize the database for the experiment.
        """
        self.db = PsychoPhysicsDatabase(self.files["database"], self.settings["Study"]["ID"])
        self.db.add_experiment(
            title=self.settings["Study"]["Title"],
            shorthand=self.settings["Study"]["Shorthand"],
            repository=self.settings["Study"]["Repository"],
            version=self.version,
            description=self.settings["Study"]["Description"],
            n_subjects=self.exp_params["NumSubjects"],
            n_sessions=self.exp_params["NumSessions"]
        )

    def init_clocks(self):
        """
        Set up the clocks for the experiment.
        """
        clocks = {
            "global": core.Clock(),
            "block": core.Clock(),
            "trial": core.Clock()
        }
        return clocks

    def init_dirs_dict(self) -> Dict:
        """
        Make a dictionary of relevant directories

        Returns
            dict: data, sub, and task keys and the corresponding Path objects as values.
        """
        dirs = dict()

        # Project
        dirs["project"] = self.root

        # Config directory
        config_dir = self.root / "config"
        dirs["config"] = config_dir

        # Data directory
        data_dir = self.root / "data" / f"v{self.version}"
        data_dir.mkdir(parents=True, exist_ok=True)
        dirs["data"] = data_dir

        # Stimuli directory
        stimuli_dir = self.root / "stimuli"
        dirs["stimuli"] = stimuli_dir

        # Log directory
        log_dir = self.root / "log" / f"v{self.version}"
        log_dir.mkdir(parents=True, exist_ok=True)
        dirs["log"] = log_dir

        return dirs

    def init_files_dict(self) -> Dict:
        """
        Make a dictionary of important files. The structure of data files follows the BIDS convention.
            Example:
                data/sub-01/ses-01/stimuli/sub-01_ses-01_task-threshold_block-01_stimuli.csv
                data/sub-01/ses-01/behavioral/sub-01_ses-01_task-threshold_block-01_choices.csv
                data/sub-01/ses-01/eyetracking/sub-01_ses-01_task-contrast_block-01_saccades.csv

        Returns
            dict: participants, stimuli, and other files as keys and the corresponding Path objects as values.
        """
        files = dict()

        # Config files
        set_file = self.directories["config"] / "settings.yaml"
        if not set_file.exists():
            raise FileNotFoundError(f"Settings file not found: {set_file}")
        files["settings"] = str(set_file)

        # Parameters files
        param_files = Path(self.directories["config"]).glob("*.yaml")
        for file in param_files:
            if file.stem != "settings":
                files[f"{file.stem}_params"] = str(file)

        # sqlite database
        db_file = self.directories["data"] / f"{self.name}_v{self.version}.db"
        if not db_file.exists():
            db_file.touch()
        files["database"] = str(db_file)

        # Participants file
        part_file = self.directories["data"] / "participants.tsv"
        if not part_file.exists():
            self.init_participants_file()
        files["participants"] = str(part_file)

        # Modules
        modules = self.self.platfrom_settings["Modules"]
        for mod in modules:
            name = mod["name"]
            path = mod["path"]
            files[name] = path

        return files

    def init_logging(self):
        """
        Generates the log file for experiment-level logging.

        Returns:
            logging.Logger: The logger object.
        """
        # Make the logger object
        logger = log_handler.get_logger(self.name)

        # Log file
        log_file = self.directories["log"] / f"exp_{self.name}_v{self.version}.log"
        self.files["log"] = str(log_file)
        if log_file.exists():
            # clear the log file
            log_file.unlink()
        # create the log file
        log_file.touch()

        # Add new handlers
        log_handler.add_handlers(logger, log_file)

        # Set the level of the handlers
        if self.debug:
            log_handler.set_handlers_level(logger.StreamHandler, logging.DEBUG)

        # First log
        logger.info("Greetings, my friend! I'm your experiment logger for the day.")
        logger.info(f"Our today's experiment is: {self.name}.")
        logger.info("Here are some settings we're working with: ")
        for key, value in self.settings["Study"].items():
            logger.info(f"\t\t -{key}: {value}")
        for key, value in self.platform_settings.items():
            logger.info(f"\t\t -{key}: {value}")
        logger.info("Let's get started!")
        logger.info("-" * 80)

        return logger

    def init_subject_dirs(self, subj_id: str, task_name: str):
        """
        Add directories for the subject.

        Args:
            subj_id (str): Subject ID
            task_name (str): Task name
        """
        # Subject directory
        subj_dir = self.directories["data"] / f"sub-{subj_id}"
        subj_dir.mkdir(parents=True, exist_ok=True)
        self.dirs["subject"] = subj_dir

        # Log directory
        log_dir = subj_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.dirs["log"] = log_dir

        # Session directory
        for ses_num in range(1, self.exp_params["NumSessions"] + 1):
            ses_dir = subj_dir / f"ses-{ses_num}"
            ses_dir.mkdir(parents=True, exist_ok=True)
            self.dirs["session"] = ses_dir

            # Behavioral directory
            beh_dir = ses_dir / "behavioral"
            beh_dir.mkdir(parents=True, exist_ok=True)
            self.dirs["behavioral"] = beh_dir

            # Stimuli directory
            stim_dir = ses_dir / "stimuli"
            stim_dir.mkdir(parents=True, exist_ok=True)
            self.dirs["stimuli"] = stim_dir

    def init_subject_files(self, subj_id: str, task_name: str):
        """
        Add files for the subject.

        Args:
            subj_id (str): Subject ID
            task_name (str): Task name
        """
        # Stimuli file
        stim_file = self.directories["stimuli"] / f"sub-{subj_id}_task-{task_name}_stimuli.csv"
        if not stim_file.exists():
            stim_file.touch()
        self.files["stimuli"] = str(stim_file)

        # Behavioral file
        beh_file = self.directories["behavioral"] / f"sub-{subj_id}_task-{task_name}_behavioral.csv"
        if not beh_file.exists():
            beh_file.touch()
        self.files["behavioral"] = str(beh_file)

    def make_display(self) -> Tuple:
        """
        Makes psychopy monitor and window objects to be used in the experiment. Relies on the name of the monitor
        specified in confing/parameters.json and monitor specification found in config/monitors.json

        Returns:
            tuple
                mon : psychopy.monitors.Monitor

                win : psychopy.visual.Window
        """
        # Make the monitor object and set width, distance, and resolution
        monitor = monitors.Monitor(
            name=self.platform_settings["Monitor"],
            width=self.monitor_params["size_cm"][0],
            distance=self.monitor_params["distance"],
            autoLog=False
        )
        monitor.setSizePix(self.monitor_params["size_px"])

        # Gamma correction
        gamma_file = self.directories["config"] / f"{self.monitor_params}_gamma_grid.npy"
        try:
            grid = np.load(str(gamma_file))
            monitor.setLineariseMethod(1)  # (a + b**xx)**gamma
            monitor.setGammaGrid(grid)
        except FileNotFoundError:
            self.logger.warning("No gamma grid file found. Running without gamma correction.")
            monitor.setGamma(None)

        # Set variables for the window object based on Debug status of the experiment
        if self.debug:
            monitor_size_px = [1200, 800]
            full_screen = False
            show_gui = True
        else:
            monitor_size_px = self.monitor_params["size_px"]
            full_screen = True
            show_gui = False

        # Make the window object
        window = visual.Window(
            name='DebugWindow' if self.debug else 'ExperimentWindow',
            monitor=monitor,
            fullscr=full_screen,
            units='deg',  # units are always degrees by default
            size=monitor_size_px,  # But size is in pixels
            allowGUI=show_gui,
            waitBlanking=True,
            color=0,  # default to mid-grey
            screen=0,  # the internal display is used by default
            autoLog=False
        )

        # Some debugging features
        if self.debug:
            window.mouseVisible = True
        else:
            window.mouseVisible = False

        return monitor, window

    def hello_gui(self):
        """
        GUI for choosing/registering subjects.

        Returns:
            str: The choice made by the user.

        Raises:
            ValueError: If the experiment is cancelled.
        """
        dlg = gui.Dlg(
            title="What do you want from me?",
            labelButtonOK="Let's go",
            labelButtonCancel="Away with you",
            alwaysOnTop=True
        )
        dlg.addText("Experiment", self.name, color="blue")
        dlg.addText("Date", data.getDateStr(), color="blue")
        dlg.addFixedField("Version", self.version)
        dlg.addField("Participant", choices=["Register", "Load"])

        dlg.show()
        if dlg.OK:
            return dlg.data[0]
        else:
            raise ValueError("Experiment cancelled.")

    def subject_registration_gui(self):
        """
        GUI for registering new subject.

        Returns:
            pd.DataFrame: The subject info as a pandas DataFrame.

        Raises:
            ValueError: If the registration is cancelled.
        """
        # Setup GUI
        dlg = gui.Dlg(
            title="Participant Registration",
            labelButtonOK="Register",
            labelButtonCancel="Go Away",
            alwaysOnTop=True
        )
        dlg.addText("Experiment", self.name, color="blue")
        dlg.addText("Date", data.getDateStr(), color="blue")
        dlg.addFixedField("Version", self.version)

        # Check what PIDs are already used
        pids = [f"{id:02d}" for id in range(1, int(self.exp_params["NumSubjects"]) + 1)]
        burnt_pids = self.load_participants_file()["PID"].values
        available_pids = [pid for pid in pids if pid not in burnt_pids]

        # Add fields
        dlg.addField("PID", choices=available_pids)
        required_info = self.exp_params["SubjectInfo"]
        for field in required_info:
            if isinstance(required_info[field], list):
                dlg.addField(field, choices=required_info[field])
            else:
                if field != "PID":
                    dlg.addField(field)
                else:
                    pass

        # Show the dialog
        dlg.show()
        if dlg.OK:
            info_dict = {"PID": dlg.data[0]}
            for i, field in enumerate(required_info.keys()):
                info_dict[field] = dlg.data[i + 1]
            info_df = pd.DataFrame([info_dict])
            return info_df
        else:
            raise ValueError("Subject registration cancelled.")

    def subject_loadation_gui(self):
        """
        GUI for loading the subject.

        Returns:
            pd.DataFrame: The subject info as a pandas DataFrame.

        Raises:
            ValueError: If the loadation is cancelled.
        """
        # Make the dialog
        dlg = gui.Dlg(
            title="Participant Loadation",
            labelButtonOK="Load",
            labelButtonCancel="Go Away",
            alwaysOnTop=True
        )
        dlg.addText("Experiment", self.name, color="blue")
        dlg.addText("Date", data.getDateStr(), color="blue")
        dlg.addFixedField("Version", self.version)

        # Find the PID and initials that are registered
        df = self.load_participants_file()
        pids = df["PID"].values
        initials = df["Initials"].values
        pid_initials = [f"{pid} ({initial})" for pid, initial in zip(pids, initials)]
        dlg.addField("PID", choices=pid_initials)

        # Show the dialog
        dlg.show()
        if dlg.OK:
            # Load the subject info
            subj_pid = dlg.data[0].split(" ")[0]
            subj_info = df.loc[df["PID"] == subj_pid, :]
            return subj_info
        else:
            raise ValueError("Subject loadation cancelled.")

    def task_selection_gui(self, subj_info: pd.DataFrame):
        """
        GUI for choosing the task.

        Args:
            subj_info (pd.DataFrame): The subject info as a pandas DataFrame.

        Returns:
            str: The task name.

        Raises:
            ValueError: If the task selection is cancelled.
        """
        # Make the dialog
        dlg = gui.Dlg(
            title="Task Selection",
            labelButtonOK="Alright",
            labelButtonCancel="Go Away",
            alwaysOnTop=True
        )
        dlg.addText("Experiment", self.name, color="blue")
        dlg.addText("Date", data.getDateStr(), color="blue")
        dlg.addFixedField("Version", self.version)

        # Add the subject info
        valid_info_fields = ["PID"] + list(self.exp_params["SubjectInfo"].keys())
        for field in subj_info.columns:
            if field in valid_info_fields:
                dlg.addFixedField(field, subj_info[field].values[0])

        # Find the tasks that are available
        tasks = list(self.exp_params["Tasks"].keys())
        burnt_tasks = subj_info.columns[subj_info.columns.str.contains("Task")].values
        available_tasks = [task for task in tasks if task not in burnt_tasks]
        dlg.addField("Task", choices=available_tasks)

        # Show the dialog
        dlg.show()
        if dlg.OK:
            return dlg.data[0]
        else:
            raise ValueError("Task selection cancelled.")

    def load_subject_info(self, pid: str):
        """
        Loads the subject info from the subject file.

        Args:
            pid (str): Participant ID

        Returns:
            pd.DataFrame: The subject info as a pandas DataFrame.

        Raises:
            ValueError: If the PID is not found in the participants file.
        """
        df = pd.read_csv(self.files["participants"], sep="\t")
        if pid in df["PID"].values:
            return df.loc[df["PID"] == pid, :]
        else:
            raise ValueError(f"PID {pid} not found in the participant file.")

    def load_participants_file(self):
        """
        Loads the participants file.

        Returns:
            pd.DataFrame: The participants file as a pandas DataFrame.
        """
        return pd.read_csv(self.files["participants"], sep="\t")

    def init_participants_file(self):
        """
        Initializes the participants file.
        """
        columns = list(self.exp_params["Info"]["Subject"].keys())
        columns += [f"{task}Task" for task in self.exp_params["Tasks"].keys()]
        columns += ["Experimenter", "ExperimentName", "Version"]
        df = pd.DataFrame(columns=columns)
        df.to_csv(self.files["participants"], sep="\t", index=False)

    def save_participants_file(self, subject_info):
        """
        Saves the subject info to the participants file.

        Args:
            subject_info (dict or DataFrame): The subject info as a dictionary.
        """
        df = pd.read_csv(self.files["participants"], sep="\t")
        if isinstance(subject_info, dict):
            sub_df = pd.DataFrame([subject_info], columns=subject_info.keys())
        out_df = pd.concat([df, sub_df], ignore_index=True)
        out_df.to_csv(self.files["participants"], sep="\t", index=False)

    def get_system_status(self, rf_thresh: float = 0.5, ram_thresh: int = 1000) -> str:
        """
        Check the status of the system, including:
            - Standard deviation of screen refresh rate
            - Amount of free RAM
            - Any interfering processes
            - Priority of the python program running the experiment
        """
        # Initial system check
        run_info = info.RunTimeInfo(
            version=self.study_settings["Version"],
            win=self.window,
            refreshTest='grating',
            userProcsDetailed=True,
            verbose=True
        )

        # Start testing
        self.logger.info("Running system checks.")
        display_warnings = ""

        # Test the refresh rate of the monitor
        rf = self.window.getActualFrameRate(nIdentical=20, nMaxFrames=100, nWarmUpFrames=10, threshold=rf_thresh)
        intended_rf = int(self.monitor_params["refresh_rate"])

        if rf is None:
            self.logger.critical("No identical frame refresh times were found.")
            display_warnings += "(✘) No identical frame refresh times. You should quit the experiment IMHO.\n\n"
        else:
            # check if the measured refresh rate is the same as the one intended
            rf = np.round(rf).astype(int)
            if rf != intended_rf:
                self.logger.warning(
                    f"The actual refresh rate {rf} does not match the intended refresh rate {intended_rf}."
                )
                display_warnings += f"(✘) The actual refresh rate {rf} does not match {intended_rf}.\n\n"
            else:
                display_warnings += "(✓) Monitor refresh rate checks out.\n\n"

        # Look for flagged processes
        flagged = run_info['systemUserProcFlagged']
        if len(flagged):
            procs = "Flagged processes: "
            display_warnings += "\t(✘) Flagged processes: "
            for pr in np.unique(flagged):
                procs += f"{pr}, "
                display_warnings += f"{pr}, "
            self.logger.warning(procs)
            display_warnings += "\n\n"
        else:
            display_warnings += "(✓) No flagged processes.\n\n"

        # See if we have enough RAM
        if run_info["systemMemFreeRAM"] < ram_thresh:
            self.logger.warning(f"Only {round(run_info['systemMemFreeRAM'] / 1000)} GB  of RAM available.")
            display_warnings += f"(✘) Only {round(run_info['systemMemFreeRAM'] / 1000)} GB  of RAM available.\n\n"
        else:
            display_warnings += "(✓) RAM is OK.\n\n"

        # Raise the priority of the experiment for CPU
        # Check if it's Mac OS X (these methods don't run on that platform)
        if self.platform_settings["OS"] in ["darwin", "Mac OS X"]:
            self.logger.warning("Cannot raise the priority because you are on Mac OS X.")
            display_warnings += "(✘) Cannot raise the priority because you are on Mac OS X.\n\n"
        else:
            try:
                Computer.setPriority("realtime", disable_gc=True)
                display_warnings += "(✓) Realtime processing is set.\n\n"
            except Exception as e:
                self.logger.warning(f"Error in elevating processing priority: {e}")
                display_warnings += f"(✘) Error in elevating processing priority: {e}.\n\n"

        return display_warnings

    def clear_screen(self):
        """ clear up the PsychoPy window"""

        # Turn off all visual stimuli
        for stim in self.stimuli.values():
            stim.autoDraw = False

        # Flip the window
        self.window.flip()

    def show_msg(self, text, wait_for_keypress: bool = True, wait_time: int = 3):
        """ Show task instructions on screen"""

        # Make a message stimulus
        msg = visual.TextStim(
            self.window,
            text,
            font="Trebuchet MS",
            color="black",
            alignText='center',
            anchorHoriz='center',
            anchorVert='center',
            wrapWidth=self.window.size[0] / 2,
            autoLog=False
        )

        # Clear the screen and show the message
        self.clear_screen()
        msg.draw()
        self.window.flip()

        # wait indefinitely, terminates upon any key press
        if wait_for_keypress:
            # Ctrl + C quits the experiment. Resume otherwise.
            pressed = event.waitKeys(modifiers=True)
            if (pressed[0] == 'c') and pressed[1]['ctrl']:
                self.goodbye()
        else:
            core.wait(wait_time)

        # Clear the screen again
        self.clear_screen()

    def goodbye(self):
        """
        Closes and ends the experiment.
        """
        self.logger.info("Bye bye experiment.")
        self.window.close()
        core.quit()
        sys.exit()

    def enable_force_quit(self, key_press: str = 'escape'):
        """
        Quits the experiment during runtime if a key (default 'space') is pressed.

        Parameters
        ----------
        key_press : str
            keyboard button to press to quit the experiment.
        """
        # Get the keypress from user
        pressed = event.getKeys()

        # Check if it's the quit key
        if key_press in pressed:
            self.logger.critical("Force quitting...")
            self.goodbye()

    def check_frame_durations(self, frame_intervals: Union[list, np.array]):
        """
        Checks the duration of all the provided frame intervals and measures the number of dropped frames.

        Args:
            frame_intervals (list, np.array): List of frame intervals

        Returns:
            int: Number of dropped frames
        """
        # Get the stats of the frames
        z_intervals = stats.zscore(frame_intervals)

        # Count anything greater than 3 standard deviations as dropped frames
        n_dropped = len(np.where(z_intervals > 3)[0])

        return n_dropped

    def ms2fr(self, duration: float):
        """ Converts durations from ms to display frames"""
        return np.ceil(duration * self.params["refresh_rate"] / 1000).astype(int)


class PsychoPhysicsDatabase:
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
