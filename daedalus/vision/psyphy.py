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

from emoji import emojize

import numpy as np
import pandas as pd

from psychopy import core, monitors, visual, event, info
from psychopy.tools.monitorunittools import deg2pix, pix2deg
import psychopy.gui.qtgui as gui
from psychopy.iohub.devices import Computer

from daedalus import log_handler, utils, codes

# from daedalus.data.database import PsychoPhysicsDatabase


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
        self.part_file = None
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
        self.all_blocks = None

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

        # Stimuli
        self.msg_stim = None

    def init_session(self):
        """
        Starts up the experiment by initializing the settings, directories, files, logging, monitor, window, and clocks.
        """
        # Initialize the settings and parameters
        self.init_configs()

        # Ask for subject information
        select, expert = self.hello_gui()
        if select == "Load":
            sub_info = self.subject_loadation_gui()
        else:
            sub_info = self.subject_registration_gui()

        # Session and task info
        ses, task = self.session_task_selection_gui(sub_info)

        # Save the information
        sub_info["PID"] = self._fix_id(sub_info["PID"].values[0])
        sub_info["Session"] = ses
        sub_info["Task"] = task
        sub_info["Experimenter"] = expert
        sub_info["Experiment"] = self.name
        sub_info["Version"] = self.version
        self.sub_id = sub_info["PID"].values[0]
        self.ses_id = self._fix_id(ses)

        # Setup the directories and files
        self.init_dirs()
        self.init_files()

        # Initialize the logging
        self.init_logging()

        # Start the database
        # self.init_database()

        # Make monitor and window
        self.init_display()

        # Make clocks
        self.global_clock = core.Clock()
        self.block_clock = core.Clock()
        self.trial_clock = core.MonotonicClock()

        # Stimuli and blocks and task
        self.task_name = sub_info["Task"].values[0]
        self.all_blocks = self.concatenated_blocks()
        self.make_msg_stim()

        # Save subject and log the start of the session
        self.update_participants(sub_info)
        self.logger.info("Greetings, my friend! I'm your experiment logger for the day.")
        self.logger.info(f"Looks like we're running a {self.exp_type} experiment called {self.name} (v{self.version}).")
        self.logger.info(f"Today is {self.today}.")
        self.logger.info(f"Subject {self.sub_id}")
        self.logger.info(f"Session {self.ses_id}")
        self.logger.info(f"Task: {self.task_name}.")
        self.logger.info("Let's get started!")
        self.logger.info("-" * 80)

    def prepare_block(self, block_id, repeat=False):
        """
        Prepare the block for the task.

        Args:
            block_id (int): Block number.
            repeat (bool): Is the block a repeat?
        """
        # Set the ID of the block
        self.block_id = self._fix_id(block_id)

        # Show the start of the block message
        n_blocks = f"{len(self.all_blocks):02d}"
        if repeat:
            txt = f"You are repeating block {self.block_id}/{n_blocks}.\n\n"
        else:
            txt = f"You are about to begin block {self.block_id}/{n_blocks}.\n\n"
        txt += "Press Space to start."
        while True:
            resp = self.show_msg(txt)
            if (resp == "space") or (self.debug):
                # Reset the block clock
                self.block_clock.reset()
                # Initialize the block data
                self.init_block_data()
                # Log the start of the block
                self.block_info(self.codex.message("block", "init"))
                self.block_info(f"BLOCKID_{self.block_id}")
                break

    def prepare_trial(self, trial_id):
        """
        Prepare the trial for the block.

        Args:
            trial_id (int): Trial number.
        """
        self.trial_id = self._fix_id(trial_id)
        self.trial_clock.reset()
        self.trial_info(self.codex.message("trial", "init"))
        self.trial_info(f"TRIALID_{self.trial_id}")

    def wrap_trial(self):
        """
        Wrap up the trial.
        """
        self.trial_info(self.codex.message("trial", "fin"))
        self.window.recordFrameIntervals = False
        self.window.frameIntervals = []
        self.clear_screen()

    def stop_trial(self):
        """
        Stop the trial.
        """
        self.stim_data.loc[int(self.trial_id) - 1, "TrialRepeated"] = 1
        self.behav_data.loc[int(self.trial_id) - 1, "TrialRepeated"] = 1
        self.trial_warning(self.codex.message("trial", "stop"))
        self.window.flip()
        self.window.recordFrameIntervals = False
        self.window.frameIntervals = []
        self.clear_screen()

    def wrap_block(self):
        """
        Wrap up the block.
        """
        # Log the end of the block
        self.block_info(self.codex.message("block", "fin"))

        # Save the data
        self.save_stim_data()
        self.save_behav_data()
        self.save_frame_data()

        # Show the end of the block message
        self.show_msg(f"Great job! You finished block {self.block_id}.", wait_time=5)
        self.clear_screen()

    def stop_block(self):
        """
        Stop the block.
        """
        self.block_warning(self.codex.message("block", "stop"))
        msg = f"Stopping block {self.block_id} due to too many failed trials ."
        self.show_msg(msg, wait_time=self.stim_params["Message"]["warning_duration"], msg_type="warning")

    def turn_off(self):
        """
        End the session.
        """
        # Log the end of the session
        self.logger.info(self.codex.message("ses", "fin"))
        msg = f"Session {self.ses_id} of the experiment is over. Thank you for participating :)"
        self.show_msg(msg, wait_time=self.stim_params["Message"]["thanks_duration"], msg_type="info")
        # Save
        self.save_session_complete()
        # Quit
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

    def save_session_complete(self):
        """
        Save the participants data.
        """
        df = self.load_participants_file()
        cond = ((df["PID"] == self.sub_id) & (df["Session"] == self.ses_id) & (df["Task"] == self.task_name))
        df.loc[cond, "Completed"] = self.today
        df.to_csv(self.part_file, sep="\t", index=False)

    def save_log_data(self):
        """
        Save the log data.
        """
        self.logger.close()

    def init_configs(self):
        """
        Initialize the settings and parameters for the experiment.
        """
        config_dir = self.root / "config"
        self.exp_params = utils.read_config(config_dir / "experiment.yaml")
        self.stim_params = utils.read_config(config_dir / "stimuli.yaml")
        monitor_name = self.settings["Platforms"][self.platform]["Monitor"]
        self.monitor_params = utils.read_config(config_dir / "monitors.yaml")[monitor_name]

    # def init_database(self):
    #     """
    #     Initialize the database for the experiment.
    #     """
    #     self.db = PsychoPhysicsDatabase(self.files["database"], self.settings["Study"]["ID"])
    #     self.db.add_experiment(
    #         title=self.settings["Study"]["Title"],
    #         shorthand=self.settings["Study"]["Shorthand"],
    #         repository=self.settings["Study"]["Repository"],
    #         version=self.version,
    #         description=self.settings["Study"]["Description"],
    #         n_subjects=self.exp_params["NumSubjects"],
    #         n_sessions=self.exp_params["NumSessions"]
    #     )

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
        self.log_dir = self.root / "log"
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
        self.log_file = self.log_dir / f"sub-{self.sub_id}_ses-{self.ses_id}_v{self.version}.log"
        if self.log_file.exists():
            self.log_file.unlink()
        self.log_file.touch()

        # sqlite database
        self.db_file = self.root / "data" / f'{self.settings["Study"]["Shorthand"]}_v{self.version}.db'
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

        # Log file
        if self.log_file.exists():
            # clear the log file
            self.log_file.unlink()
        # create the log file
        self.log_file.touch()

        # Make the logger
        self.logger = log_handler.DaedalusLogger(self.name, self.debug, self.log_file)

    def block_debug(self, msg):
        """
        Log a debug message for the block.

        Args:
            msg (str): The message to be logged.
        """
        self.logger.debug(msg, extra={"block": self.block_id}, stack_info=True, stacklevel=4)

    def block_info(self, msg):
        """
        Log a message for the trial.

        Args:
            msg (str): The message to be logged.
        """
        self.logger.info(msg, extra={"block": self.block_id}, stack_info=True, stacklevel=4)

    def block_warning(self, msg):
        """
        Log a warning message for the trial.

        Args:
            msg (str): The message to be logged.
        """
        self.logger.warning(msg, extra={"block": self.block_id}, stack_info=True, stacklevel=4)

    def block_error(self, msg):
        """
        Log an error message for the trial.

        Args:
            msg (str): The message to be logged.
        """
        self.logger.error(msg, extra={"block": self.block_id}, stack_info=True, stacklevel=4)

    def block_critical(self, msg):
        """
        Log a critical message for the trial.

        Args:
            msg (str): The message to be logged.
        """
        self.logger.critical(msg, extra={"block": self.block_id}, stack_info=True, stacklevel=4)

    def trial_debug(self, msg):
        """
        Log a debug message for the trial.

        Args:
            msg (str): The message to be logged.
        """
        self.logger.debug(msg, extra={"block": self.block_id, "trial": self.trial_id}, stack_info=True, stacklevel=4)

    def trial_info(self, msg):
        """
        Log a message for the trial.

        Args:
            msg (str): The message to be logged.
        """
        self.logger.info(msg, extra={"block": self.block_id, "trial": self.trial_id}, stack_info=True, stacklevel=4)

    def trial_warning(self, msg):
        """
        Log a warning message for the trial.

        Args:
            msg (str): The message to be logged.
        """
        self.logger.warning(msg, extra={"block": self.block_id, "trial": self.trial_id}, stack_info=True, stacklevel=4)

    def trial_error(self, msg):
        """
        Log an error message for the trial.

        Args:
            msg (str): The message to be logged.
        """
        self.logger.error(msg, extra={"block": self.block_id, "trial": self.trial_id}, stack_info=True, stacklevel=4)

    def trial_critical(self, msg):
        """
        Log a critical message for the trial.

        Args:
            msg (str): The message to be logged.
        """
        self.logger.critical(msg, extra={"block": self.block_id, "trial": self.trial_id}, stack_info=True, stacklevel=4)

    def _fix_id(self, id_):
        """
        Fix the ID of the subject or session to be a string with two digits.

        Args:
            id_ (int): The ID to be fixed.

        Returns:
            str: The fixed ID.
        """
        if isinstance(id_, (int, np.integer)):
            id_ = f"{id_:02d}"
        elif isinstance(id_, str) and len(id_) == 1:
            id_ = f"0{id_}"
        elif isinstance(id_, (list, np.ndarray)):
            return [self._fix_id(i) for i in id_]
        return id_

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
        files = self.ses_data_dir.glob(
            f"sub-{self.sub_id}_ses-{self.ses_id}_task-{self.task_name}_block-*_behavioral.csv"
        )
        comp_blocks = [file.stem.split("_")[-2].split("-")[1] for file in files]
        remain_blocks = [block for block in all_blocks if block not in comp_blocks]

        return remain_blocks

    def init_frames_data(self):
        """
        Initialize the frame intervals data.
        """
        fname = f"sub-{self.sub_id}_ses-{self.ses_id}_task-{self.task_name}_block-{self.block_id}_FrameIntervals.csv"
        frames_file = self.ses_data_dir / fname
        if frames_file.exists():
            self.block_warning(f"File {frames_file} already exists. Renaming the file as backup.")
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
        fname = f"sub-{self.sub_id}_ses-{self.ses_id}_task-{self.task_name}_block-{self.block_id}_stimuli.csv"
        stim_file = self.ses_data_dir / fname
        if stim_file.exists():
            self.block_warning(f"File {stim_file} already exists. Renaming the file as backup.")
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
        fname = f"sub-{self.sub_id}_ses-{self.ses_id}_task-{self.task_name}_block-{self.block_id}_behavioral.csv"
        behav_file = self.ses_data_dir / fname
        if behav_file.exists():
            self.block_warning(f"File {behav_file} already exists. Renaming the file as backup.")
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

    def load_session_data(self, data_type):
        """
        Load the session data.
        """
        files = self.ses_data_dir.glob(
            f"sub-{self.sub_id}_ses-{self.ses_id}_task-{self.task_name}_block-*_{data_type}.csv"
        )
        data = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)

        return data

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
            gamma_file = self.root / "config" / f"{name}_gamma_grid.npy"
            try:
                grid = np.load(str(gamma_file))
                monitor.setLineariseMethod(1)  # (a + b*xx)**gamma
                monitor.setGammaGrid(grid)
            except FileNotFoundError:
                self.logger.warning("No gamma grid file found. Running without gamma correction.")
                monitor.setGamma(None)

            # Save for future use
            # monitor.save()

        # Set variables for the window object based on Debug status of the experiment
        if self.debug:
            monitor_size_px = [1200, 800]
            full_screen = False
            show_gui = True
        else:
            monitor_size_px = self.monitor_params["size_pix"]
            full_screen = True
            show_gui = False

        # Make the window object
        window = visual.Window(
            name='DebugWindow' if self.debug else 'ExperimentWindow',
            monitor=monitor,
            fullscr=full_screen,
            units='pix',
            size=monitor_size_px,
            allowGUI=show_gui,
            waitBlanking=True,
            color=0,  # default to mid-grey
            screen=0,  # the internal display is used by default
            checkTiming=False,
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
        emos = self.stim_params["Emojis"]
        dlg = gui.Dlg(
            title="Yo!",
            font=self.stim_params["Display"]["gui_font"],
            font_size=self.stim_params["Display"]["gui_font_size"],
            size=int(self.stim_params["Display"]["gui_size"]),
            labelButtonOK=emojize(f"Let's Go {emos['lets_go']}"),
            labelButtonCancel=emojize(f"Go Away {emos['go_away']}"),
            alwaysOnTop=True
        )
        # Add the experiment info
        color = utils.str2tuple(self.stim_params["Display"]["gui_text_color"])
        dlg.addText("Experiment", color=color)
        dlg.addFixedField(key="title", label=emojize(f"{emos['title']}Title"), initial=emojize(f"{self.name}"))
        dlg.addFixedField(key="date", label=emojize(f"{emos['date']}Date"), initial=self.today)
        dlg.addFixedField(key="version", label=emojize(f"{emos['version']}Version"), initial=self.version)
        dlg.addFixedField(
            key="debug",
            label=emojize(f"{emos['debug']} Debug Mode"),
            # initial=emojize("On :check_box_with_check:") if self.debug else emojize("Off :cross_mark:")
            initial="On" if self.debug else "Off"
        )

        # Participant and experimenter
        dlg.addText(
            emojize("Who are you?"),
            color=color
        )
        field = "experimenter"
        dlg.addField(
            key=field,
            label=emojize(f"{emos[field]} {field.capitalize()}"),
            choices=[emojize(a["name"]) for a in self.settings["Study"]["Authors"]]
        )
        dlg.addField(
            key="selection",
            label=emojize(f"{emos['participant']} Participant"),
            choices=[emojize("Register"), emojize("Load")]
        )

        info = dlg.show()
        if dlg.OK:
            return info["selection"], info["experimenter"]
        else:
            raise SystemExit("Experiment cancelled.")

    def subject_registration_gui(self):
        """
        GUI for registering new subject.

        Returns:
            pd.DataFrame: The subject info as a pandas DataFrame.

        Raises:
            ValueError: If the registration is cancelled.
        """
        # Setup GUI
        emos = self.stim_params["Emojis"]
        dlg = gui.Dlg(
            title=f"{self.name} Experiment",
            font=self.stim_params["Display"]["gui_font"],
            font_size=self.stim_params["Display"]["gui_font_size"],
            size=self.stim_params["Display"]["gui_size"],
            labelButtonOK=emojize(f"Register {emos['register']}"),
            labelButtonCancel=emojize(f"Go Away {emos['go_away']}"),
            alwaysOnTop=True
        )

        # Check what PIDs are already used
        color = utils.str2tuple(self.stim_params["Display"]["gui_text_color"])
        dlg.addText(
            emojize(f"{emos['participant']} Participant"),
            color=color
        )
        min_pid, max_pid = self.exp_params["Subjects"]["PID"].split("-")
        pids = [f"{i:02d}" for i in range(int(min_pid), int(max_pid) + 1)]
        pdf = self.load_participants_file()
        burnt_pids = self._fix_id(pdf["PID"].values)
        available_pids = []
        for pid in pids:
            if pid not in burnt_pids:
                available_pids.append(pid)

        # Add fields
        dlg.addField(key="PID", label=emojize(f"{emos['pid']} PID"), choices=available_pids)
        required_info = self.exp_params["Subjects"]
        for field, value in required_info.items():
            field_key = field.replace("_", "")
            field_name = field.replace("_", " ")
            em_key = field.lower()
            if isinstance(value, list):
                dlg.addField(key=field_key, label=emojize(f"{emos[em_key]} {field_name}"), choices=value)
            elif isinstance(value, str):
                if field != "PID":
                    if "-" in value:
                        min_, max_ = value.split("-")
                        range_vals = np.arange(int(min_), int(max_)+1).tolist()
                        dlg.addField(key=field_key, label=emojize(f"{emos[em_key]} {field_name}"), choices=range_vals)
                    else:
                        dlg.addField(key=field_key, label=emojize(f"{emos[em_key]} {field_name}"), initial=value)

        # Show the dialog
        info = dlg.show()
        if dlg.OK:
            for col in pdf.columns:
                if col not in info.keys():
                    info[col] = None
            sub_df = pd.DataFrame.from_dict({k: [v] for k, v in info.items()}, orient="columns")
            return sub_df
        else:
            raise SystemExit("Subject registration cancelled.")

    def subject_loadation_gui(self):
        """
        GUI for loading the subject.

        Returns:
            pd.DataFrame: The subject info as a pandas DataFrame.

        Raises:
            ValueError: If the loadation is cancelled.
        """
        # Make the dialog
        emos = self.stim_params["Emojis"]
        dlg = gui.Dlg(
            title="Participant Loadation",
            font=self.stim_params["Display"]["gui_font"],
            font_size=self.stim_params["Display"]["gui_font_size"],
            size=self.stim_params["Display"]["gui_size"],
            labelButtonOK=emojize(f"Load {emos['load']}"),
            labelButtonCancel=emojize(f"Go Away {emos['go_away']}"),
            alwaysOnTop=True
        )

        # Find the PID and initials that are registered
        color = utils.str2tuple(self.stim_params["Display"]["gui_text_color"])
        dlg.addText(
            emojize(f"{emos['participant']} Participant"),
            color=color
        )
        df = self.load_participants_file()
        pids = self._fix_id(df["PID"].values)
        initials = df["Initials"].values
        pid_initials = [f"{pid} ({initial})" for pid, initial in zip(pids, initials)]
        dlg.addField(key="sub", label=emojize(f"{emos['pid']} PID (Initials)"), choices=pid_initials)

        # Show the dialog
        info = dlg.show()
        if dlg.OK:
            # Load the subject info
            pid = info["sub"].split(" ")[0]
            sub_df = df.loc[df["PID"].astype(int) == int(pid), :]
            return sub_df
        else:
            raise SystemExit("Subject loadation cancelled.")

    def session_task_selection_gui(self, sub_df):
        """
        GUI for choosing the session.

        Args:
            sub_df (DataFrame): The subject information.

        Returns:
            int: The session number.

        Raises:
            ValueError: If the session selection is cancelled.
        """
        # Make the dialog
        emos = self.stim_params["Emojis"]
        dlg = gui.Dlg(
            title=f"{self.name} Experiment",
            font=self.stim_params["Display"]["gui_font"],
            font_size=self.stim_params["Display"]["gui_font_size"],
            size=self.stim_params["Display"]["gui_size"],
            labelButtonOK=emojize(f"Alright {emos['alright']}"),
            labelButtonCancel=emojize(f"Go Away {emos['go_away']}"),
            alwaysOnTop=True
        )
        # Add the subject info
        color = utils.str2tuple(self.stim_params["Display"]["gui_text_color"])
        dlg.addText(
            emojize(f"{emos['participant']} Participant Information"),
            color=color
        )
        for field in sub_df.columns:
            for valid in self.exp_params["Subjects"].keys():
                valid_key = valid.replace("_", "")
                valid_emo = valid.lower().replace(" ", "_")
                label = emojize(f"{emos[valid_emo]} {valid.replace('_', ' ')}")
                if field == valid_key:
                    dlg.addFixedField(
                        key=valid_key,
                        label=label,
                        initial=sub_df[field].values[0]
                    )

        # Add sessions and tasks that are available
        complete = sub_df.loc[~(pd.isnull(sub_df["Completed"])), ["Session", "Task"]]
        choices = []
        for ses in self.exp_params["Sessions"]:
            if ses["id"] not in complete["Session"].values:
                for task in ses["tasks"]:
                    choices.append(f"{ses['id']} ({task})")
            else:
                burnt_tasks = complete.loc[complete["Session"] == ses["id"], "Task"].values
                available_tasks = [task for task in ses["tasks"] if task not in burnt_tasks]
                for task in available_tasks:
                    choices.append(f"{ses['id']} ({task})")
        dlg.addField(
            key="choice",
            label=emojize(f"{emos['session']} Session (Task)"),
            choices=choices,
        )

        # Show the dialog
        info = dlg.show()
        if dlg.OK:
            ses, task = info["choice"].split(" ")
            ses = self._fix_id(ses)
            task = task[1:-1]
            return ses, task
        else:
            raise SystemExit("Session selection cancelled.")

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
        if self.part_file is None:
            data_dir = Path(self.root) / "data" / f"v{self.version}"
            if not data_dir.exists():
                data_dir.mkdir(parents=True, exist_ok=True)
            self.part_file = data_dir / "participants.tsv"
        if not self.part_file.exists():
            df = self._init_part_file()
        else:
            df = pd.read_csv(self.part_file, sep="\t")

        return df

    def _init_part_file(self):
        """
        Initializes the participants file.
        """
        columns = ["Session", "Task", "Completed", "Experimenter", "Experiment", "Version"]
        columns += [k.replace("_", "") for k in self.exp_params["Subjects"].keys()]
        df = pd.DataFrame(columns=columns, dtype=str)
        df["Completed"] = pd.to_datetime(df["Completed"])
        df.to_csv(self.part_file, sep="\t", index=False)

        return df

    def update_participants(self, sub_info):
        """
        Saves the subject info to the participants file.

        Args:
            subject_info (dict or DataFrame): The subject info as a dictionary.
        """
        if isinstance(sub_info, dict):
            info = {k: [v] for k, v in sub_info.items()}
            sub_df = pd.DataFrame.from_dict(info, orient="columns")
        else:
            sub_df = sub_info

        pdf = self.load_participants_file()
        pids = self._fix_id(pdf["PID"].values)
        self.logger.debug(f"Loaded PIDs: {pdf['PID'].values}")
        self.logger.debug(f"Available PIDs: {pids}")
        if self.sub_id not in pids:
            df = pd.concat([pdf, sub_df], ignore_index=False)
            df.to_csv(self.part_file, sep="\t", index=False)
        else:
            duplicate = pdf.loc[
                (pdf["PID"] == self.sub_id) & (pdf["Session"] == self.ses_id) & (pdf["Task"] == self.task_name)
            ]
            if duplicate.shape[0] == 0:
                df = pd.concat([pdf, sub_df], ignore_index=False)
                df.to_csv(self.part_file, sep="\t", index=False)

    def add_to_participant(self, col, value):
        """
        Updates the participant information.

        Args:
            sub_info (dict): The subject information.
        """
        # df = self.load_participants_file()
        df = pd.read_csv(self.part_file, sep="\t")
        cond = ((df["PID"] == self.sub_id) & (df["Session"] == self.ses_id) & (df["Task"] == self.task_name))
        df.loc[cond, col] = value
        df.to_csv(self.part_file, sep="\t", index=False)

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
        results = []
        self.logger.info("Running system checks.")
        results.append("<Press Space to accept or Escape to quit>\n\n")

        # Refresh rate
        rf = self.window.getActualFrameRate(nIdentical=20, nMaxFrames=100, nWarmUpFrames=10, threshold=rf_thresh)
        intended_rf = int(self.monitor_params["refresh_rate"])
        if rf is None:
            self.logger.critical("No identical frame refresh times were found.")
            results.append("(✘) No identical frame refresh times. You should quit the experiment IMHO.")
        else:
            # check if the measured refresh rate is the same as the one intended
            rf = np.round(rf).astype(int)
            if rf != intended_rf:
                self.logger.warning(
                    f"The actual refresh rate {rf} does not match the intended refresh rate {intended_rf}."
                )
                results.append(f"(✘) The actual refresh rate {rf} does not match {intended_rf}.")
            else:
                self.logger.info("Monitor refresh rate checks out.")
                results.append("(✓) Monitor refresh rate checks out.")

        # Processes
        flagged = run_info['systemUserProcFlagged']
        if flagged:
            results.append("(✘) Flagged processes:")
            w = ""
            for proc in np.unique(flagged):
                w += f"\t- {proc}\n"
            self.logger.warning(f"Flagged processes: {w}")
            results.append(w)
        else:
            self.logger.info("No flagged processes.")
            results.append("(✓) No flagged processes.")

        # See if we have enough RAM
        if run_info["systemMemFreeRAM"] < ram_thresh:
            self.logger.warning(f"Only {round(run_info['systemMemFreeRAM'] / 1000)} GB  of RAM available.")
            results.append(f"(✘) Only {round(run_info['systemMemFreeRAM'] / 1000)} GB  of RAM available.")
        else:
            self.logger.info("RAM is OK.")
            results.append("(✓) RAM is OK.")

        # Raise the priority of the experiment for CPU
        # Check if it's Mac OS X (these methods don't run on that platform)
        if self.settings["Platforms"][self.platform]["OS"] in ["darwin", "Mac OS X"]:
            self.logger.warning("Cannot raise the priority because you are on Mac OS X.")
            results.append("(✘) Cannot raise the priority because you are on Mac OS X.")
        else:
            try:
                Computer.setPriority("realtime", disable_gc=True)
                results.append("(✓) Realtime processing is set.")
                self.logger.info("Realtime processing is set.")
            except Exception as e:
                results.append(f"(✘) Error in elevating processing priority: {e}.")
                self.logger.warning(f"Error in elevating processing priority: {e}")

        return results

    def clear_screen(self):
        """ clear up the PsychoPy window"""

        # Reset background color
        self.window.color = utils.str2tuple(self.stim_params["Display"]["background_color"])
        # Flip the window
        self.window.flip()

    def make_msg_stim(self):
        """Make a message stimulus"""

        # Make a message stimulus
        self.msg_stim = visual.TextStim(
            self.window,
            font="Trebuchet MS",
            height=self.stim_params["Message"]["font_height"],
            color=utils.str2tuple(self.stim_params["Message"]["normal_text_color"]),
            alignText='left',
            anchorHoriz='right',
            anchorVert='center',
            wrapWidth=self.window.size[0] / 3,
            autoLog=False
        )

    def make_countdown_stim(self):
        """Make a countdown stimulus"""

        # Make a countdown stimulus
        params = self.stim_params["Message"]
        countdown = visual.TextStim(
            self.window,
            font=params["font"],
            color=utils.str2tuple(params["normal_text_color"]),
            alignText='center',
            anchorHoriz='center',
            anchorVert='center',
            wrapWidth=self.window.size[0] / 2,
            pos=(self.window.size[0] / 2, self.window.size[1] / 2),
            autoLog=False
        )

        return countdown

    def show_msg(self, text, wait_keys=True, wait_time=0, msg_type="info"):
        """
        Show task instructions on screen

        Args:
            text (str): The text to be displayed.
            wait_keys (bool): Whether to wait for a key press.
            wait_time (float): The time to wait before continuing.

        Returns:
            str: The key press that was made.
        """
        # Clear the screen
        self.clear_screen()

        # Change background color
        params = self.stim_params["Message"]
        self.window.color = utils.str2tuple(params["background_color"])
        self.window.flip()

        # Set the text
        self.msg_stim.text = text

        # Change color based on info
        if msg_type == "info":
            self.msg_stim.color = utils.str2tuple(params["normal_text_color"])
        elif msg_type in ["warning", "error"]:
            self.msg_stim.color = utils.str2tuple(params["warning_text_color"])

        # Show the message
        self.msg_stim.draw()
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
            self.goodbye(self.codex.message("usr", "term"))
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
            self.logger.critical(self.codex.message("exp", "term"))
            try:
                # Save as much as you can
                self.save_stim_data()
                self.save_behav_data()
                self.save_frame_data()
                self.save_log_data()
            except AttributeError as e:
                self.logger.error(f"Error in saving data: {e}")
                raise SystemExit(f"Experiment ended with error: {raise_error}")
        else:
            # Log
            self.logger.info("Bye Bye Experiment.")
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

    def concatenated_blocks(self):
        """
        Concatenates the blocks for the task.
        """
        conc_blocks = []
        for block in self.exp_params["Tasks"][self.task_name]["blocks"]:
            for _ in range(block["n_blocks"]):
                conc_blocks.append(block)
        return conc_blocks

    def ms2fr(self, duration: float):
        """
        Converts durations from ms to display frames:
            duration (ms) * 1/1000 (s/ms) * rf (frames/s) = n (frames)

        Args:
            duration (float): Duration in ms.

        Returns:
            int: Duration in frames
        """
        return np.ceil(duration * (1/1000) * self.monitor_params["refresh_rate"]).astype(int)

    def fr2ms(self, n: int):
        """
        Converts durations from display frames to ms:
            n (frames) * 1/rf (s/frame) * 1000 (ms/s) = duration (ms)

        Args:
            n (int): Duration in frames.

        Returns:
            float: Duration in ms.
        """
        return n * (1/self.monitor_params["refresh_rate"]) * 1000

    def hz2cpf(self, cps: float):
        """
        Converts cycles per second to cycles per frame:
            cps (cycles/s) * 1/rf (s/frames) = cpf (cycles/frame)

        Args:
            cps (float): Cycles per second (temporal frequency).

        Returns:
            float: Cycles per frame
        """
        return cps * (1/self.monitor_params["refresh_rate"])

    def hz2dpf(self, cps: float, cpd: float):
        """
        Converts cycles per second to degrees per frame:
            cps (cycles/s) * 1/cpd (dva/cycles) * 1/rf (s/frame) = dpf (dva/frame)

        Args:
            cps (float): Cycles per second (temporal frequency).
            cpd (float): Cycles per degree (spatial frequency).

        Returns:
            float: Degrees per frame.
        """
        return cps * (1/cpd) * (1/self.monitor_params["refresh_rate"])

    def hz2dps(self, cps: float, cpd: float):
        """
        Converts cycles per second to degrees per second:
            cps (cycles/s) * 1/cpd (dva/cycles) = dps (dva/s)

        Args:
            cps (float): Cycles per second (temporal frequency).
            cpd (float): Cycles per degree (spatial frequency).

        Returns:
            float: Degrees per second.
        """
        return cps * (1/cpd)

    def hz2pps(self, cps: float, cpd: float):
        """
        Converts cycles per second to pixels per second:
            cps (cycles/s) * 1/cpd (dva/cycles) * ppd (pix/dva) = pps (pix/s)

        Args:
            cps (float): Cycles per second (temporal frequency).
            cpd (float): Cycles per degree (spatial frequency).

        Returns:
            float: Pixels per second.
        """
        ppd = deg2pix(1, self.monitor)
        return cps * (1/cpd) * ppd

    def hz2ppf(self, cps: float, cpd: float):
        """
        Converts cycles per second to pixels per frame:
            cps (cycles/s) * 1/cpd (dva/cycles) * ppd (pix/dva) * 1/rf (s/frame) = ppf (pix/frame)

        Args:
            cps (float): Cycles per second (temporal frequency).
            cpd (float): Cycles per degree (spatial frequency).

        Returns:
            float: Pixels per frame.
        """
        ppd = deg2pix(1, self.monitor)
        return cps * (1/cpd) * ppd * (1/self.monitor_params["refresh_rate"])

    def cpd2cpp(self, cpd: float):
        """
        Converts cycles per degree to cycles per pixel:
            cpd (cycles/dva) * dpp (dva/pixel) = cpp (cycles/pixel)

        Args:
            cpd (float): Cycles per degree (spatial frequency).

        Returns:
            float: Cycles per pixel.
        """
        dpp = pix2deg(1, self.monitor)
        return cpd * dpp

    def cpp2cpd(self, cpp: float):
        """
        Converts cycles per pixel to cycles per degree:
            cpp (cycles/pixel) * ppd (pixel/dva) = cpd (cycles/dva)

        Args:
            cpp (float): Cycles per pixel.

        Returns:
            float: Cycles per degree.
        """
        ppd = deg2pix(1, self.monitor)
        return cpp * ppd

    def time_point_to_frame_idx(self, time, frame_times):
        """
        Find the frame number for a given time.
        """
        return np.argmax(np.cumsum(frame_times) > time)

    @staticmethod
    def period_edge_frames(frames, period):
        """
        Find the start and end frames of a period in a frame array.
        """
        modified = np.where(frames == period, 100, 0)
        start = np.argmax(modified)
        end = frames.shape[0] - 1 - np.argmax(modified[::-1])

        return start, end

    def val_round(self, val):
        return np.round(val, self.exp_params["General"]["round_decimal"])

    def ms_round(self, times):
        return np.round(times * 1000, self.exp_params["General"]["round_decimal"])
