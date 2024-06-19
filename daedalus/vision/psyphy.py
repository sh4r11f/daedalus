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
from pathlib import Path
from typing import Dict, Union, Tuple
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

from psychopy import core, monitors, visual, event, info, gui
from psychopy.iohub.devices import Computer

import logging
from daedalus import log_handler, utils, codes

from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from daedalus.data.database import (
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
    def __init__(self, root: Union[str, Path], platform: str, debug: bool):

        # Setup
        self.exp_type = 'psychophysics'
        self.root = Path(root)
        self.platform = platform
        self.debug = debug
        self.today = date.today().strftime("%b-%d-%Y")

        # Settings
        self.settings = utils.read_config(self.root / "config" / "settings.yaml")
        self.name = self.settings["Study"]["Title"]
        self.version = self.settings["Study"]["Version"]

        # Configs
        self.exp_params = None
        self.stim_params = None
        self.monitor_params = None

        # Directories and files
        self.data_dir = None
        self.log_dir = None
        self.ses_data_dir = None
        self.db_file = None
        self.participants_file = None
        self.log_file = None
        self.stim_data_file = None
        self.frames_data_file = None
        self.behav_data_file = None

        # Data
        self.db = None
        self.stim_data = None
        self.frames_data = None
        self.behav_data = None

        # Subject, session, block, and trial
        self.sub_id = None
        self.ses_id = None
        self.block_id = None
        self.trial_id = None

        # Task
        self.task_name = None
        self.task = None

        # Log
        self.logger = None
        self.codex = codes.Codex()

        # Visuals
        self.monitor = None
        self.window = None

        # Clocks
        self.global_clock = None
        self.block_clock = None
        self.trial_clock = None

    def init_session(self):
        """
        """
        # Initialize the settings and parameters
        self.init_configs()

        # Ask for subject information
        hello = self.hello_gui()
        if hello["selection"] == "Load":
            sub_info = self.subject_loadation_gui()
        else:
            sub_info = self.subject_registration_gui()
        self.sub_id = self._fix_id(sub_info["PID"])

        # Session info
        ses_info = self.session_task_selection_gui()
        self.ses_id = self._fix_id(ses_info["ses_id"])
        self.task_name = ses_info["task"]

        # Add extra info and save
        sub_info["Experimenter"] = hello["experimenter"]
        sub_info["Session"] = self.ses_id
        sub_info["Task"] = self.task_name
        self.save_subject_info(sub_info)

        # Setup the directories and files
        self.init_dirs()
        self.init_files()

        # Initialize the logging
        self.init_logging()

        # Start the database
        self.init_database()

        # Make monitor and window
        self.init_display()

        # Make clocks
        self.global_clock = core.Clock()
        self.block_clock = core.Clock()
        self.trial_clock = core.MonotonicClock()

    def prepare_block(self):
        """
        Prepare the block for the task.
        """
        # Start the clock and log
        self.block_clock.reset()
        self.log_info(self.codex.message("block", "init"))
        self.log_info(f"BLOCKID_{self.block_id}")

        # Initialize the block data
        self.init_block_data()

        # Show the start of the block message
        n_blocks = len(self.get_reamining_blocks())
        txt = f"You are about to begin block {self.block_id}/{n_blocks}.\n\n"
        txt += "Press Space to start."
        resp = self.show_msg(txt)

        if resp == "space":
            msg = self.codex.message("exp", "ok")
            self.log_info(msg)
        elif resp in ["escape", "ctrl+c"]:
            self.goodbye(self.codex.message("exp", "term"))

    def prepare_trial(self):
        """
        Prepare the trial for the block.

        Args:
            trial_num (int): Trial number.
        """
        self.trial_clock.reset()
        self.log_info(self.codex.message("trial", "init"))
        self.log_info(f"TRIALID_{self.trial_id}")

    def wrap_trial(self):
        """
        Wrap up the trial.
        """
        self.log_info(self.codex.message("trial", "fin"))
        self.window.recordFrameIntervals = False
        self.window.frameIntervals = []
        self.clear_screen()

    def wrap_block(self, block_name):
        """
        Wrap up the block.
        """
        # Log the end of the block
        self.log_info(self.codex.message("block", "fin"))

        # Save the data
        self.save_stim_data()
        self.save_behav_data()
        self.save_frame_data()

        # Show the end of the block message
        self.show_msg(f"Great job! You finished block {self.block_id}.", wait_time=5)
        self.clear_screen()

    def end_session(self):
        """
        End the session.
        """
        self.log_info(self.codex.message("ses", "fin"))
        self.show_msg(f"Amazing! You finished session {self.ses_id} of the experiment", wait_time=5)
        self.clear_screen()

        # Save and quit
        self.save_session_complete()
        self.goodbye()

    def save_stim_data(self):
        """
        Save the stimulus data.
        """
        self.stim_data.to_csv(self.stim_data_file, sep=",", index=False)

    def save_behav_data(self):
        """
        Save the behavioral data.
        """
        self.behav_data.to_csv(self.behav_data_file, sep=",", index=False)

    def save_frame_data(self):
        """
        Save the frame data.
        """
        self.frames_data.to_csv(self.frames_data_file, sep=",", index=False)

    def save_session_complete(self, experimenter):
        """
        Save the participants data.
        """
        sdf = self.load_subject_info()
        df = sdf.loc[(sdf["Session"] == self.ses_id) & (sdf["Task"] == self.task_name)]
        df["Completed"] = self.today
        df.to_csv(self.participants_file, sep="\t", index=False)

    def save_log_data(self):
        """
        Save the log data.
        """
        self.logger.handlers[0].close()

    def init_configs(self):
        """
        Initialize the settings and parameters for the experiment.
        """
        config_dir = self.root / "config"
        self.exp_params = utils.read_config(config_dir / "experiment.yaml")
        self.stim_params = utils.read_config(config_dir / "stimuli.yaml")
        monitor_name = self.settings["Platforms"][self.platform]["Monitor"]
        self.monitor_params = utils.read_config(config_dir / "monitors.yaml")[monitor_name]

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

    def init_dirs(self) -> Dict:
        """
        Make a dictionary of relevant directories

        Returns
            dict: data, sub, and task keys and the corresponding Path objects as values.
        """
        # Data directory
        self.data_dir = self.root / "data" / f"v{self.version}"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Log directory
        self.log_dir = self.root / "log" / f"v{self.version}"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Subject directory
        sub_data_dir = self.data_dir / f"sub-{self.sub_id}"
        sub_data_dir.mkdir(parents=True, exist_ok=True)

        # Session directory
        self.ses_data_dir = sub_data_dir / f"ses-{self.ses_id}"
        self.ses_data_dir.mkdir(parents=True, exist_ok=True)

    def init_files(self) -> Dict:
        """
        Make a dictionary of important files. The structure of data files follows the BIDS convention.
            Example:
                data/sub-01/ses-01/stimuli/sub-01_ses-01_task-threshold_block-01_stimuli.csv
                data/sub-01/ses-01/behavioral/sub-01_ses-01_task-threshold_block-01_choices.csv
                data/sub-01/ses-01/eyetracking/sub-01_ses-01_task-contrast_block-01_saccades.csv

        Returns
            dict: participants, stimuli, and other files as keys and the corresponding Path objects as values.
        """
        # Log file
        self.log_file = self.log_dir / f"sub-{self.sub_id}_ses-{self.ses_id}.log"
        if self.log_file.exists():
            self.log_file.unlink()
        self.log_file.touch()

        # sqlite database
        self.db_file = self.data_dir / f"{self.name}.db"
        if not self.db_file.exists():
            self.db_file.touch()

        # Participants file
        self.part_file = self.data_dir / "participants.tsv"
        if not self.part_file.exists():
            self._init_part_file()

    def init_logging(self):
        """
        Generates the log file for experiment-level logging.

        Returns:
            logging.Logger: The logger object.
        """
        # Make the logger object
        self.logger = log_handler.get_logger(self.name)

        # Log file
        if self.log_file.exists():
            # clear the log file
            self.log_file.unlink()
        # create the log file
        self.log_file.touch()

        # Add new handlers
        log_handler.add_handlers(self.logger, self.log_file)

        # Set the level of the handlers
        if self.debug:
            log_handler.set_handlers_level(self.logger.StreamHandler, logging.DEBUG)

        # First log
        self.logger.info("Greetings, my friend! I'm your experiment logger for the day.")
        self.logger.info("Here are some settings we're working with: ")
        for key, value in self.settings["Study"].items():
            self.logger.info(f"\t\t -{key}: {value}")
        for key, value in self.platform_settings.items():
            self.logger.info(f"\t\t -{key}: {value}")
        self.logger.info("Let's get started!")
        self.logger.info("-" * 80)

    def log_info(self, msg):
        """
        Log an info message and add the name of the function that called it.

        Args:
            msg (str): The message to be logged.
        """
        self.logger.info(f"[@{utils.get_caller()}] {msg}")

    def log_warning(self, msg):
        """
        Log a warning message and add the name of the function that called it.

        Args:
            msg (str): The message to be logged.
        """
        self.logger.warning(f"[@{utils.get_caller()}] {msg}")

    def log_error(self, msg):
        """
        Log an error message and add the name of the function that called it.

        Args:
            msg (str): The message to be logged.
        """
        self.logger.error(f"[@{utils.get_caller()}] {msg}")

    def log_critical(self, msg):
        """
        Log a critical message and add the name of the function that called it.

        Args:
            msg (str): The message to be logged.
        """
        self.logger.critical(f"[@{utils.get_caller()}] {msg}")

    def _fix_id(self, id_):
        """
        Fix the ID of the subject or session to be a string with two digits.

        Args:
            id_ (int): The ID to be fixed.

        Returns:
            str: The fixed ID.
        """
        if isinstance(id_, int):
            fixed = f"{id_:02d}"
        elif isinstance(id_, str) and len(id_) == 1:
            fixed = f"0{id_}"
        return fixed

    def _get_ses_tasks(self):
        """
        Get the tasks for the session.

        Returns:
            list: List of tasks for the session.
        """
        return utils.find_in_configs(self.exp_params["Tasks"], "Session", self.ses_id)["tasks"]

    def _get_task_blocks(self):
        """
        Get the blocks for the task.

        Returns:
            list: List of blocks for the task.
        """
        return self.exp_params["Tasks"][self.task_name]["blocks"]

    def get_remaining_blocks(self):
        """
        Check the completed blocks for the task.

        Returns:
            list: List of completed blocks for the task.
        """
        all_blocks = self._get_task_blocks()
        files = self.ses_data_dir.glob(f"sub-{self.sub_id}_ses-{self.ses_id}_task-{self.task_name}_block-*_behavioral.csv")
        comp_blocks = [file.stem.split("_")[-2].split("-")[1] for file in files]
        remain_blocks = [block for block in all_blocks if block not in comp_blocks]

        return remain_blocks

    def init_frames_data(self):
        """
        Initialize the frame intervals data.
        """
        frames_file = self.ses_data_dir / f"sub-{self.sub_id}_ses-{self.ses_id}_task-{self.task_name}_block-{self.block_id}_FrameIntervals.csv"
        if frames_file.exists():
            self.log_warning(f"File {frames_file} already exists. Renaming the file as backup.")
            frames_file.rename(frames_file.with_suffix(".BAK"))
        self.frames_data_file = frames_file
        self.frames_data = pd.DataFrame(columns=[
            "TrialIndex",
            "Period",
            "FrameIndex", "FrameDuration_ms", "Frame_TrialTime_ms"
        ])

    def init_stim_data(self):
        """
        Initialize the stimulus data.
        """
        stim_file = self.ses_data_dir / f"sub-{self.sub_id}_ses-{self.ses_id}_task-{self.task_name}_block-{self.block_id}_stimuli.csv"
        if stim_file.exists():
            self.log_warning(f"File {stim_file} already exists. Renaming the file as backup.")
            stim_file.rename(stim_file.with_suffix(".BAK"))
        self.stim_data_file = stim_file
        self.stim_data = pd.DataFrame(columns=[
            "BlockID", "BlockName", "BlockDuration_sec",
            "TrialIndex", "TrialNumber", "TrialDuration_ms", "TrialDuration_fr",
        ])

    def init_behav_data(self):
        """
        Initialize the behavioral data.
        """
        behav_file = self.ses_data_dir / f"sub-{self.sub_id}_ses-{self.ses_id}_task-{self.task_name}_block-{self.block_id}_behavioral.csv"
        if behav_file.exists():
            self.log_warning(f"File {behav_file} already exists. Renaming the file as backup.")
            behav_file.rename(behav_file.with_suffix(".BAK"))
        self.behav_data_file = behav_file
        self.behav_data = pd.DataFrame(columns=[
            "BlockID", "BlockName", "BlockDuration_sec",
            "TrialIndex", "TrialNumber", "TrialDuration_ms", "TrialDuration_fr",
        ])

    def init_block_data(self):
        """
        Initialize the block data.
        """
        self.init_stim_data()
        self.init_behav_data()
        self.init_frames_data()

    def init_display(self) -> Tuple:
        """
        Makes psychopy monitor and window objects to be used in the experiment. Relies on the name of the monitor
        specified in confing/parameters.json and monitor specification found in config/monitors.json

        Returns:
            tuple
                mon : psychopy.monitors.Monitor

                win : psychopy.visual.Window
        """
        name = self.settings["Platforms"][self.platform]["Monitor"]
        available = monitors.getAllMonitors()
        if name in available:
            monitor = monitors.Monitor(name=name)
        else:
            # Make the monitor object and set width, distance, and resolution
            monitor = monitors.Monitor(
                name=name,
                width=self.monitor_params["size_cm"][0],
                distance=self.monitor_params["distance"],
                autoLog=False
            )
            monitor.setSizePix(self.monitor_params["size_pix"])

            # Gamma correction
            gamma_file = self.directories["config"] / f"{self.monitor_params}_gamma_grid.npy"
            try:
                grid = np.load(str(gamma_file))
                monitor.setLineariseMethod(1)  # (a + b**xx)**gamma
                monitor.setGammaGrid(grid)
            except FileNotFoundError:
                self.logger.warning("No gamma grid file found. Running without gamma correction.")
                monitor.setGamma(None)

            # Save for future use
            monitor.save()

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

        # Return the monitor and window objects
        self.monitor = monitor
        self.window = window

    def hello_gui(self):
        """
        GUI for choosing/registering subjects.

        Returns:
            str: The choice made by the user.

        Raises:
            ValueError: If the experiment is cancelled.
        """
        dlg = gui.Dlg(
            title=f"Welcome!",
            labelButtonOK="Let's go",
            labelButtonCancel="Away with you",
            alwaysOnTop=True
        )
        # Add the experiment info
        dlg.addText("Experiment", color=self.exp_params["General"]["gui_text_color"])
        dlg.addFixedField(key="title", label="Title", value=self.name)
        dlg.addFixedField(key="date", label="Date", value=self.today)
        dlg.addFixedField(key="version", label="Version", value=self.version)
        dlg.addFixedField(key="debug", label="Debug Mode", value="On" if self.debug else "Off")

        # Participant and experimenter
        dlg.addText("Who are you?", color=self.exp_params["General"]["gui_text_color"])
        dlg.addField(key="experimenter", value="Experimenter", choices=[a["name"] for a in self.settings["Study"]["Authors"]])
        dlg.addField(key="selection", label="Participant", choices=["Register", "Load"])

        info = dlg.show()
        if dlg.OK:
            return info
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
            title=f"{self.name} Experiment",
            labelButtonOK="Register",
            labelButtonCancel="Go Away",
            alwaysOnTop=True
        )

        # Check what PIDs are already used
        dlg.addText("Participant", color=self.exp_params["General"]["gui_text_color"])
        pids = [f"{id:02d}" for id in range(1, int(self.exp_params["NumSubjects"]) + 1)]
        burnt_pids = self.load_participants_file()["PID"].values
        available_pids = [pid for pid in pids if pid not in burnt_pids]

        # Add fields
        dlg.addField(key="subj_id", label="PID", choices=available_pids)
        required_info = self.exp_params["SubjectInfo"]
        for field in required_info:
            if isinstance(required_info[field], list):
                dlg.addField(key=field, label=field,  choices=required_info[field])
            elif isinstance(required_info[field], str):
                if "-" in required_info[field]:
                    min_, max_ = required_info[field].split("-")
                    dlg.addField(key=field, label=field, choices=list(range(int(min_), int(max_)+1)))
            else:
                if field != "PID":
                    dlg.addField(key=field, label=field)
                else:
                    pass

        # Show the dialog
        info = dlg.show()
        if dlg.OK:
            return info
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

        # Find the PID and initials that are registered
        dlg.addText("Participant", color=self.exp_params["General"]["gui_text_color"])
        df = self.load_participants_file()
        pids = df["PID"].values
        initials = df["Initials"].values
        pid_initials = [f"{pid} ({initial})" for pid, initial in zip(pids, initials)]
        dlg.addField(key="PID", label="ID (Initials)", choices=pid_initials)

        # Show the dialog
        info = dlg.show()
        if dlg.OK:
            # Load the subject info
            choice = info["PID"]
            pid = choice.split(" ")[0]
            initials = choice.split(" ")[1][1:-1]
            return {"PID": pid, "Initials": initials}
        else:
            raise ValueError("Subject loadation cancelled.")

    def session_task_selection_gui(self):
        """
        GUI for choosing the session.

        Returns:
            int: The session number.

        Raises:
            ValueError: If the session selection is cancelled.
        """
        # Make the dialog
        dlg = gui.Dlg(
            title=f"{self.name} Experiment",
            labelButtonOK="Alright",
            labelButtonCancel="Go Away",
            alwaysOnTop=True
        )
        # Add the subject info
        dlg.addText("Participant Information", color=self.exp_params["General"]["gui_text_color"])
        valid_info_fields = ["PID"] + list(self.exp_params["SubjectInfo"].keys())
        sub_info = self.load_subject_info()
        for field in sub_info.columns:
            if field in valid_info_fields:
                dlg.addFixedField(key=field, label=field, value=sub_info[field].values[0])

        # Add sessions and tasks that are available
        dlg.addText("Session and Task", color=self.exp_params["General"]["gui_text_color"])
        comp = sub_info.loc[~(pd.isnull(sub_info["Completed"])), ["Session", "Task"]]
        for ses in self.exp_params["Sessions"]:
            if ses["id"] not in comp["Session"].values:
                dlg.addField(key=f"ses_{ses['id']}", label=f"Session {ses['id']}", choices=ses["tasks"])
            else:
                burnt_tasks = comp.loc[comp["Session"] == ses["id"], "Task"].values
                available_tasks = [task for task in ses["tasks"] if task not in burnt_tasks]
                dlg.addField(key=f"ses_{ses['id']}", label=f"Session {ses['id']}", choices=available_tasks)

        # Show the dialog
        info = dlg.show()
        if dlg.OK:
            ses = [key.split("_")[1] for key in info.keys() if "ses" in key][0]
            task = info[f"ses_{ses}"]
            return {"ses": ses, "task": task}
        else:
            raise ValueError("Session selection cancelled.")

    def load_subject_info(self):
        """
        Loads the subject info from the subject file.

        Returns:
            pd.DataFrame: The subject info as a pandas DataFrame.

        Raises:
            ValueError: If the PID is not found in the participants file.
        """
        pdf = self.load_participants_file()
        if self.sub_id in pdf["PID"].values:
            return pdf.loc[pdf["PID"] == self.sub_id, :]
        else:
            raise ValueError(f"PID {self.sub_id} not found in the participant file.")

    def load_participants_file(self):
        """
        Loads the participants file.

        Returns:
            pd.DataFrame: The participants file as a pandas DataFrame.
        """
        return pd.read_csv(self.participants_file, sep="\t")

    def _init_participants_file(self):
        """
        Initializes the participants file.
        """
        columns = ["PID", "Session", "Task", "Completed", "Experimenter", "Experiment", "Version"]
        columns += list(self.exp_params["SubjectInfo"].keys())
        df = pd.DataFrame(columns=columns)
        df.to_csv(self.participants_file, sep="\t", index=False)

    def save_subject_info(self, sub_info):
        """
        Saves the subject info to the participants file.

        Args:
            subject_info (dict or DataFrame): The subject info as a dictionary.
        """
        if isinstance(sub_info, dict):
            sub_df = pd.DataFrame.from_dict(sub_info, orient="columns")
        else:
            sub_df = sub_info

        df = self.load_participants_file()
        existing_pids = df["PID"].unique()
        if self.sub_id not in existing_pids:
            # new subject
            out_df = pd.concat([df, sub_df], ignore_index=True)
        else:
            out_df = df.copy()

        # Save the file
        out_df.to_csv(self.participants_file, sep="\t", index=False)

    def system_check(self, rf_thresh: float = 0.5, ram_thresh: int = 1000) -> str:
        """
        Check the status of the system, including:
            - Standard deviation of screen refresh rate
            - Amount of free RAM
            - Any interfering processes
            - Priority of the python program running the experiment
        """
        # Initial system check
        run_info = info.RunTimeInfo(
            version=self.settings["Study"]["Version"],
            win=self.window,
            refreshTest='grating',
            userProcsDetailed=True,
            verbose=True
        )

        # Start testing
        self.logger.info("Running system checks.")
        warnings = "<Press Space to continue or Escape to quit>\n\n"

        # Test the refresh rate of the monitor
        rf = self.window.getActualFrameRate(nIdentical=20, nMaxFrames=100, nWarmUpFrames=10, threshold=rf_thresh)
        intended_rf = int(self.monitor_params["refresh_rate"])

        if rf is None:
            self.logger.critical("No identical frame refresh times were found.")
            warnings += "(✘) No identical frame refresh times. You should quit the experiment IMHO.\n\n"
        else:
            # check if the measured refresh rate is the same as the one intended
            rf = np.round(rf).astype(int)
            if rf != intended_rf:
                self.logger.warning(
                    f"The actual refresh rate {rf} does not match the intended refresh rate {intended_rf}."
                )
                warnings += f"(✘) The actual refresh rate {rf} does not match {intended_rf}.\n\n"
            else:
                warnings += "(✓) Monitor refresh rate checks out.\n\n"

        # Look for flagged processes
        flagged = run_info['systemUserProcFlagged']
        if len(flagged):
            procs = "Flagged processes: "
            warnings += "\t(✘) Flagged processes: "
            for pr in np.unique(flagged):
                procs += f"{pr}, "
                warnings += f"{pr}, "
            self.logger.warning(procs)
            warnings += "\n\n"
        else:
            warnings += "(✓) No flagged processes.\n\n"

        # See if we have enough RAM
        if run_info["systemMemFreeRAM"] < ram_thresh:
            self.logger.warning(f"Only {round(run_info['systemMemFreeRAM'] / 1000)} GB  of RAM available.")
            warnings += f"(✘) Only {round(run_info['systemMemFreeRAM'] / 1000)} GB  of RAM available.\n\n"
        else:
            warnings += "(✓) RAM is OK.\n\n"

        # Raise the priority of the experiment for CPU
        # Check if it's Mac OS X (these methods don't run on that platform)
        if self.platform_settings["OS"] in ["darwin", "Mac OS X"]:
            self.logger.warning("Cannot raise the priority because you are on Mac OS X.")
            warnings += "(✘) Cannot raise the priority because you are on Mac OS X.\n\n"
        else:
            try:
                Computer.setPriority("realtime", disable_gc=True)
                warnings += "(✓) Realtime processing is set.\n\n"
            except Exception as e:
                self.logger.warning(f"Error in elevating processing priority: {e}")
                warnings += f"(✘) Error in elevating processing priority: {e}.\n\n"

        return warnings

    def clear_screen(self):
        """ clear up the PsychoPy window"""

        # Reset background color
        self.window.color = self.exp_params["General"]["background_color"]
        # Flip the window
        self.window.flip()

    def show_msg(self, text, wait_keys=True, wait_time=0):
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

        # Wait indefinitely, terminates upon any key press or timeout
        if wait_keys:
            valid_keys = [
                ("enter", None),
                ("space", None),
                ("escape", None),
                ("c", "ctrl")
            ]
            if isinstance(wait_keys, list):
                if all(isinstance(k, tuple) for k in wait_keys):
                    valid_keys += wait_keys

        key_press = None
        while True:
            # Check for key presses
            if wait_keys:
                pressed = event.getKeys(modifiers=True)
                for pk, mods in pressed:
                    pms = [mod for mod, val in mods.items() if val]
                    for vk, vm in valid_keys:
                        if vm is None:
                            if pk == vk:
                                key_press = pk
                        else:
                            if (pk == vk) and (vm in pms):
                                key_press = "+".join([vm, vk])
                # Break if a key is pressed
                if key_press is not None:
                    break
            # Wait for a certain amount of time
            if wait_time:
                core.wait(wait_time)
                break

        if key_press == "ctrl+c":
            self.goodbye("User quit.")
        else:
            self.clear_screen()
            return key_press

    def goodbye(self, raise_error=None):
        """
        Closes and ends the experiment.
        """
        # Close the window
        self.clear_screen()
        self.window.close()

        # Quit
        if raise_error is not None:
            # Save as much as you can
            self.log_critical(self.codex.message("exp", "term"))
            self.save_stim_data()
            self.save_behav_data()
            self.save_frame_data()
            self.save_log_data()
            raise SystemExit(f"Experiment ended with error: {raise_error}")
        else:
            # Log
            self.log_info("Bye Bye Experiment.")
            core.quit()

    def enable_force_quit(self):
        """
        Quits the experiment during runtime if Ctrl+C is pressed.
        """
        # Get the keypress from user
        pressed = event.getKeys(modifiers=True)
        # Check if it's the quit key
        if pressed:
            for key, mods in pressed:
                if key == "c" and mods["ctrl"]:
                    self.goodbye("User quit.")

    def ms2fr(self, duration: float):
        """Converts durations from ms to display frames"""
        return np.ceil(self.monitor_params["refresh_rate"] * duration / 1000).astype(int)

    def fr2ms(self, frames: int):
        """Converts durations from display frames to ms"""
        return np.ceil(frames * 1000 / self.monitor_params["refresh_rate"]).astype(int)

    def time_point_to_frame_idx(self, time, frame_times):
        """
        Find the frame number for a given time.
        """
        sum_frames = np.cumsum(frame_times)
        return np.argmax(sum_frames > time)

    @staticmethod
    def period_edge_frames(frames, period):
        modified = np.where(frames == period, 100, 0)
        start = np.argmax(modified, axis=1)
        end = np.argmax(modified[::-1], axis=1)

        return start, end

    def val_round(self, val):
        return np.round(val, self.exp_params["General"]["round_decimal"])

    def ms_round(self, times):
        return np.round(times * 1000, self.exp_params["General"]["round_decimal"])


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
