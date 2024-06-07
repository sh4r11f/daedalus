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


class Psychophysics:
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
        self.platform_settings = self.settings["Platforms"][self.platform]
        self.exp_params = utils.read_config(self.files["experiment_params"])
        self.stim_params = utils.read_config(self.files["stimuli_params"])
        monitor_name = self.platfrom_settings["Monitor"]
        mon_params = utils.read_config(self.files["monitors"])
        self.monitor_params = mon_params[monitor_name]

        # Other attributes
        self.logger = self.init_logging()
        self.monitor, self.window = self.make_display()
        self.clocks = self.init_clocks()

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

    def init_directories(self) -> Dict:
        """
        Sets up the directories and paths that are important in the experiment. The structure roughly follows the
        BIDS convention.

        Example:  data/FIPS/sub-01/staircase/
                        data/FIPS2/sub-02/perceptual/
                        ...
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

    def init_files(self) -> Dict:
        """
        Sets up the files that are important in the experiment. The structure roughly follows the BIDS convention.

        Example:  data/FIPS/sub-01/staircase/
                        data/FIPS2/sub-02/perceptual/
                        ...
        Returns
            dict: participants, stimuli, and other files as keys and the corresponding Path objects as values.
        """
        files = dict()

        # Config files
        files["settings"] = str(self.directories["config"] / "settings.yaml")

        # Parameters files
        param_files = Path(self.directories["config"]).glob("*.yaml")
        for file in param_files:
            if file != "settings.yaml":
                files[f"{file.stem}_params"] = str(file)

        # Participants file
        files["participants"] = str(self.directories["data"] / "participants.tsv")
        if not Path(files["participants"]).exists():
            self.init_participants_file()

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

    def make_display(self) -> Tuple:
        """
        Makes psychopy monitor and window objects to be used in the experiment. Relies on the name of the monitor
        specified in confing/parameters.json and monitor specification found in config/monitors.json

        Returns
        -------
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
        required_info = self.exp_params["Info"]["Subject"]
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

    def choose_task(self, subj_info: pd.DataFrame):
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
        valid_info_fields = ["PID"] + list(self.exp_params["Info"]["Subject"].keys())
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
        # Make a quick check here
        file = self.files.get("participants")
        if file is None:
            self.files["participants"] = self.directories["data"] / "participants.tsv"
        else:
            if not file.exists():
                self.init_participants_file()
            else:
                return pd.read_csv(file, sep="\t")

    def init_participants_file(self):
        """
        Initializes the participants file.
        """
        columns = list(self.exp_params["Info"]["Subject"].keys())
        columns += [f"{task}Task" for task in self.exp_params["Tasks"].keys()]
        columns += ["Experimenter", "ExperimentName", "Version"]
        df = pd.DataFrame(columns=columns)
        df.to_csv(self.files["participants"], sep="\t", index=False)

    def save_subject_info(self, subject_info):
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
