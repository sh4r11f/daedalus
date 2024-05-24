# -*- coding: utf-8 -*-
# ==================================================================================================== #
#                                                                                                      #
#                                                                                                      #
#               SCRIPT: base.py                                                                        #
#                                                                                                      #
#                                                                                                      #
#          DESCRIPTION: Basic class for experiments                                                    #
#                                                                                                      #
#                                                                                                      #
#                 RULE: DAYW                                                                           #
#                                                                                                      #
#                                                                                                      #
#                                                                                                      #
#              CREATOR: Sharif Saleki                                                                  #
#                 TIME: 05-22-2024-7810598105114117                                                    #
#                SPACE: Dartmouth College, Hanover, NH                                                 #
#                                                                                                      #
# ==================================================================================================== #
import yaml
import argparse
from pathlib import Path
import json
from abc import abstractmethod, ABC
import platform
from typing import Dict, List, Union, Tuple

import pandas as pd
import numpy as np

from psychopy import gui, data, core, monitors, info, logging, visual, event
from psychopy.iohub.devices import Computer
from psychopy.tools.monitorunittools import deg2pix
import pylink

from . import Experiment

class PsychophysicsExperiment(Experiment, ABC):
    """
    Base class for all the experiments. It contains all the necessary methods and attributes that are common to all
    """
    def __init__(self, name: str, root: Union[str, Path], version: float, debug: bool, info: dict = None):

        # Setup
        self.exp_type = 'psychophysics'

        # Experiment and subject info
        if info is None:
            if self.debug:
                self.info = self.bag_debug_info()
            else:
                self.info = self.bag_info_gui()
        else:
            self.info = info

        # Set up directories and files
        self.directories = self._setup_directories()
        self.files = dict()

        # Initialize other params
        self.stimuli = dict()

        # Get monitor and make window
        self.monitor, self.window = self.make_display()

        # Set up logging
        self._init_logging()

        # Rounding numbers to these many decimal points
        self.round_err = 8

        # Clock to keep track of timing.
        self.clock = core.Clock()

    def debug_info(self):
        """
        Fake info for debugging 
        """
        info = {
            "id": 6969,
            "init": "FAKE",
            "age": 6969,
            "gender": "NULL",
            "handedness": "UP",
            "vision": "ASTRONOMICAL"
        }

        return info

    def load_config(self, conf_name: str):
        """
        Reads and saves parameters specified in a json file under the config directory, e.g. config/params.json

        Args:
            conf_name (str): Name of the file to be loaded.

        Returns:
            dict: The .json file is returned as a python dictionary.
        """
        # Find the parameters file
        params_file = self.root / "config" / f"{conf_name}.yaml"

        # Check its status and read it
        if params_file.is_file():
            try:
                with open(params_file) as pf:
                    params = yaml.safe_load(pf)
            # for corrupted files or other issues
            except IOError as e:
                logging.error(f"Unable to open the {conf_name} file: {e}")
                raise f"Unable to open the {conf_name} file: {e}"
        else:
            raise FileNotFoundError(f"{str(params_file)} does not exist")

        return params

    def bag_info_cl(self):
        """
        Get subject info from the command line when GUI is unavailable.
        """
        parser = argparse.ArgumentParser(description=f"{self.name} experiment information")
        parser.add_argument("-session", type=int, help="Session number")
        parser.add_argument("-id", type=int, help="Subject ID")
        parser.add_argument("-initials", type=str, help="Subject initials")
        parser.add_argument("-age", type=int, help="Age")
        parser.add_argument("-gender", type=str, help="Participant gender")
        parser.add_argument("-hand", type=str, help="Handedness")
        parser.add_argument("-vision", type=str, help="Vision")
        args = parser.parse_args()

        info = {
            "session": args.session,
            "id": args.id,
            "initials": args.initial,
            "age": args.age,
            "gender": args.gender,
            "handedness": args.hand,
            "vision": args.vision
        }

        return info

    def bag_info_gui(self):
        """
        Shows a gui to get the session number from user.
        """
        n_sessions = self.settings["sessions"]
        info_dict = {
            "Experiment": self.NAME,
            "Date": data.getDateStr(),
            "Session Number": list(range(1, n_sessions + 1)),
        }

        # Make the gui
        sess_gui = gui.DlgFromDict(
            title="Choose session",
            dictionary=info_dict,
            order=["Experiment", "Date", "Session Number"],
            fixed=["Experiment", "Date"],
            show=False
        )

        # Show it and get the response
        _sess = 0
        sess_gui.show()
        if sess_gui.OK:
            _sess = int(info_dict["Session Number"])
        else:
            self.end(win=False)

        # log
        logging.exp(f"Running session {_sess}")

        return _sess

    def register_sub_gui(self):
        """
        Shows a GUI to register a new subject for the experiment.
        """
        # Get previous info
        # Find the participants file
        pfile = self.HOME / "data" / "participants.tsv"

        # Load all the subjects
        _df = pd.read_csv(pfile, sep='\t')
        sub_ids = _df["ID"].tolist()  # compile IDs into a list
        sub_inits = _df["INITIALS"].tolist()

        # Initiate GUI
        sub_dlg = gui.Dlg(title="New Participant", labelButtonOK="Register", labelButtonCancel="Quit")

        # Info about the experiment
        sub_dlg.addText(text="Experiment", color="blue")
        sub_dlg.addFixedField("Title", self.NAME)
        sub_dlg.addFixedField("Date", str(data.getDateStr()))

        # Info about the subject
        sub_dlg.addText(text="Participant info", color="blue")
        sub_dlg.addField("ID", choices=list(range(1, 30)))
        sub_dlg.addField("Initials", tip="Uppercase letters (e.g. GOD)")
        sub_dlg.addField("Age", choices=list(range(18, 81)))
        sub_dlg.addField("Gender", choices=["Male", "Female", "Other"])
        sub_dlg.addField("Handedness", choices=["Right", "Left"])
        sub_dlg.addField("Vision", choices=["Normal", "Corrected"])

        # Show the dialogue
        sub_info = sub_dlg.show()

        # Get button press
        if sub_dlg.OK:

            # Experiment run and session for a new subject will be 1
            self.session = 1
            self.run = 1

            # Subject
            if sub_info[2] not in sub_ids:
                self._SUBJECT["id"] = sub_info[2]
            else:
                raise ValueError("A subject with this ID already exists.")

            if sub_info[3] not in sub_inits:
                self._SUBJECT["init"] = sub_info[3]
            else:
                raise ValueError("A subject with these initials already exists. Try changing them.")

            self._SUBJECT["age"] = int(sub_info[4])
            self._SUBJECT["gender"] = sub_info[5]
            self._SUBJECT["hand"] = sub_info[6]
            self._SUBJECT["vision"] = sub_info[7]

            # Log subject number
            logging.info(f"Subject ID: {sub_info[2]}")

        # If cancelled
        else:
            self.end(win=False)

    def load_sub_gui(self):
        """
        Loads info about subject IDs from data/participants.tsv and lets the user choose a subject for the session/run.
        """
        # Find the participants file
        pfile = self.HOME / "data" / "participants.tsv"

        # Load all the subjects
        _df = pd.read_csv(pfile, sep='\t')
        sub_ids = _df["ID"].tolist()  # compile IDs into a list
        sub_ids.sort()
        print(sub_ids)

        # Make a GUI
        sub_dlg = gui.Dlg(title="Load Participant", labelButtonOK="Load", labelButtonCancel="Quit")

        # Info about the experiment
        sub_dlg.addText(text="Experiment", color="blue")
        sub_dlg.addFixedField("Title", self.NAME)
        sub_dlg.addFixedField("Date", str(data.getDateStr()))

        # Info about the subject IDs
        sub_dlg.addField("Participant ID", choices=sub_ids)

        # Show
        _info = sub_dlg.show()

        # Get button press
        if sub_dlg.OK:

            # Convert types
            _id = int(_info[-1])
            _df["ID"] = _df["ID"].astype(int)

            # Select dataframe
            sub_df = _df[_df["ID"] == _id]

            # Load info about this subject
            self._SUBJECT["id"] = _id
            self._SUBJECT["init"] = sub_df["INITIALS"]
            self._SUBJECT["age"] = sub_df["AGE"]
            self._SUBJECT["gender"] = sub_df["GENDER"]
            self._SUBJECT["hand"] = sub_df["HAND"]
            self._SUBJECT["vision"] = sub_df["VISION"]

            # Log
            logging.info(f"Subject ID #{_id} loaded.")

        # If cancel was pressed, quit
        else:
            self.end(win=False)

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
        # Find the monitor name and specification
        mon_name = self.params["MONITOR"]
        mon_specs = self.load_config('monitors')[mon_name]
        # refresh rate is used everywhere to convert ms to frames. it's useful to have it as a parameter
        self.params["refresh_rate"] = mon_specs["refresh_rate"]

        # Make the monitor object and set width, distance, and resolution
        mon = monitors.Monitor(
            name=mon_name,
            width=mon_specs["size_cm"][0],
            distance=mon_specs["dist"],
            autoLog=False
        )
        mon.setSizePix(mon_specs["size_px"])

        # Gamma correction
        gamma_file = self.paths["config"] / f"{mon_name}_gamma_grid.npy"
        try:
            grid = np.load(str(gamma_file))
            mon.setLineariseMethod(1)  # (a + b**xx)**gamma
            mon.setGammaGrid(grid)

        except FileNotFoundError:
            print("No gamma grid file found. Running without gamma correction.")
            mon.setGamma(None)

        # Set variables for the window object based on Debug status of the experiment
        if self.DEBUG:
            mon_size = [1600, 900]
            mon_size = mon_specs["size_px"]
            full_screen = False
            # full_screen = True
            _gui = True
        else:
            mon_size = mon_specs["size_px"]
            full_screen = True
            _gui = False

        # Make the window object
        win = visual.Window(
            name='exp_window',
            monitor=mon,
            fullscr=full_screen,
            units='deg',  # units are always degrees by default
            size=mon_size,
            allowGUI=_gui,
            waitBlanking=True,
            color=self.params["BG_COLOR"],
            screen=0,  # the internal display is used by default
            autoLog=False
        )

        return mon, win

    def validate_all_frames(self, frame_intervals: Union[list, np.array]):
        """
        Checks the duration of all the provided frame intervals and measures the number of dropped frames.
        If there are too many dropped frames, throws a runtime error.

        Parameters
        ----------
        frame_intervals : list or numpy.array
            All the frame durations
        """
        # Get parameters
        mon_name = self.params["MONITOR"]
        mon_specs = self.load_config('monitors')[mon_name]
        rf = int(mon_specs["refresh_rate"])

        # Set thresholds
        thresh = .05  # threshold is at 5% of the frames
        n_thresh = len(frame_intervals) * thresh

        # Get the stats of the frames
        dur_sd = np.std(frame_intervals)
        # rf_cutoff = (1 / rf) * 1.10  # give 10% wiggle room
        rf_cutoff = (1 / rf) + (2 * dur_sd)  # frames greater than 2 std are dropped frames

        # find the dropped frames
        n_dropped = np.count_nonzero(np.array(frame_intervals) > rf_cutoff)

        # Throw an error if too many dropped frames
        if n_dropped > n_thresh and not self.DEBUG:
            # Log it
            self._log_run(f"!!! {n_dropped} dropped frames.")

            # Change fixation color to red
            self.stimuli["fix"].color = [1, -1, -1]
            self.stimuli["fix"].draw()
            self.window.flip()

            raise RuntimeError
        else:
            # Log
            self._log_run("Refresh rates OK.")

            # Report the number of dropped frames
            return n_dropped

    def get_system_status(self) -> str:
        """
        Check the status of the system, including:
            - Standard deviation of screen refresh rate
            - Amount of free RAM
            - Any interfering processes
            - Priority of the python program running

        Returns
        -------
        str
            Message about all system status to be displayed on the screen
        """
        # Somewhere to save the warnings to
        warns = ""

        # initial system check
        _run_info = info.RunTimeInfo(
            author=self.settings["author"],
            version=self.settings["version"],
            win=self.window,
            refreshTest='grating',
            userProcsDetailed=True,
            verbose=True
        )

        # Start logging
        logging.info("------------------------------------------------")
        logging.info("SYSTEM CHECKS")
        logging.info(f"Author: {_run_info['experimentAuthor']}")
        logging.info(f"Version: {_run_info['experimentAuthVersion']}")
        warns += "SYSTEM CHECKS\n"
        warns += "================================================================================================\n\n"

        # Test the refresh rate of the monitor
        rf = self.window.getActualFrameRate(nIdentical=20, nMaxFrames=100, nWarmUpFrames=10, threshold=.5)
        mon_name = self.params["MONITOR"]
        mon_specs = self.load_config('monitors')[mon_name]
        intended_rf = int(mon_specs["refresh_rate"])

        # in case could not get a good measurement
        if rf is None:
            logging.info("No identical frame refresh times were found.")
            warns += "(✘) No identical frame refresh times were found.\n\n"

        # if the measurement was successful
        else:

            # check if the measured refresh rate is the same as the one intended
            rf = np.round(rf).astype(int)

            if rf != intended_rf:
                logging.info(f"The actual refresh rate {rf} does not match the intended refresh rate {intended_rf}.")
                warns += f"(✘) The actual refresh rate {rf} does not match the intended refresh rate {intended_rf}.\n\n"

            # if everything checks out
            else:
                warns += f"(✓) Monitor refresh rate checks out.\n\n"

        # Check the monitor refresh time variability
        refresh_sd = _run_info["windowRefreshTimeSD_ms"]
        thresh = 0.20  # refresh rate standard deviation threshold
        if refresh_sd > thresh:

            # Log
            refresh_sd = np.round(refresh_sd, self.round_err)
            logging.info(f"Monitor refresh rate is too unreliable. SD: {refresh_sd}.")
            warns += f"(✘) Monitor refresh rate is too unreliable. SD: {refresh_sd}.\n\n"

            # Look for processes that might be causing the high refresh SD
            flagged = _run_info['systemUserProcFlagged']
            if len(flagged):
                s = "Flagged processes: "
                warns += "\t(✘) Flagged processes: "
                # app_set = {}
                # for i in range(len(_run_info['systemUserProcFlagged']) - 1):
                #     if _run_info['systemUserProcFlagged'][i][0] not in app_set:
                #         app_set.update({_run_info['systemUserProcFlagged'][i][0]: 0})
                # while len(app_set) != 0:
                for pr in np.unique(flagged):
                    # Log
                    # pr = app_set.popitem()[0]
                    s += f"{pr}, "
                    warns += f"{pr}, "
                warns += "\n\n"
                logging.info(s)
        else:
            warns += "(✓) Refresh threshold is normal.\n\n"

        # See if we have enough RAM
        ram_thresh = 1000
        if _run_info["systemMemFreeRAM"] < ram_thresh:
            logging.info(f"Only {round(_run_info['systemMemFreeRAM'] / 1000)} GB  of RAM available.")
            warns += f"(✘) Only {round(_run_info['systemMemFreeRAM'] / 1000)} GB  of RAM available.\n\n"
        else:
            warns += "(✓) RAM is OK.\n\n"

        # Raise the priority of the experiment for CPU
        # Check if it's Mac OS X (these methods don't run on that platform)
        if platform == "darwin":
            logging.info("Cannot raise the priority because you are on Mac OS X.")
            warns += "(✘) Cannot raise the priority because you are on Mac OS X.\n\n"
        else:
            try:
                Computer.setPriority("realtime", disable_gc=True)
                warns += "(✓) Realtime processing is set.\n\n"
            except Exception as e:
                logging.info(f"Error in elevating processing priority: {e}")
                warns += f"(✘) Error in elevating processing priority: {e}.\n\n"

        return warns

    def save_sub(self):
        """
        Saves the current newly registered subject to the data/participants.tsv file.
        """
        # Make a dictionary with current subject's parameters
        sub_info = {
            "ID": self._SUBJECT["id"],
            "INITIALS": self._SUBJECT["init"],
            "GENDER": self._SUBJECT["gender"],
            "AGE": self._SUBJECT["age"],
            "HAND": self._SUBJECT["hand"],
            "VISION": self._SUBJECT["vision"],
            f"SESSION_{self.session}": data.getDateStr()
        }

        # Read the participants file
        part_file = self.HOME / "data" / "participants.tsv"

        try:
            # Read the file
            part_df = pd.read_csv(part_file, sep='\t')

            # Check if this subject already exists
            _subs = part_df["INITIALS"].tolist()

            if self._SUBJECT["init"] in _subs:

                # If it's the first session and the subject already exists then it must be a duplicate name
                if int(self.session) == 1:
                    print("A participant with these initials already exist. Try changing the initials.")
                    logging.warning("A participant with these initials already exist. Try changing the initials.")

                # If the session is above one (and the subject exists) then just update the session number
                else:
                    logging.exp("adding to subject sessions.")
                    part_df.loc[part_df["INITIALS"] == self._SUBJECT["init"], f"SESSION_{self.session}"] = data.getDateStr()

            # If this is a new subject save the information in a new row
            else:
                # Add subject dictionary to the dataframe as a row
                logging.exp("saving new subject info.")
                part_df = part_df.append(sub_info, ignore_index=True)

        # Maybe this is the first run of the experiment ever
        except FileNotFoundError:
            # Make a new dataframe
            part_df = pd.DataFrame(sub_info)

        # Save back to the file again
        try:
            part_df.to_csv(part_file, sep='\t', index=False)
        except (FileExistsError, PermissionError) as e:
            print(f"Error in saving the participants file: {e}")
            logging.error(f"Error in saving the participants file: {e}")

    def save_exp_data(self):
        """ For saving experiment-level logs and data"""

        # self.logs["exp"].write(self.files["exp_log"])
        # self.logs["error"].write(self.files["error_log"])
        pass

    def clear_screen(self):
        """ clear up the PsychoPy window"""

        # Turn off all visual stimuli
        for stim in self.stimuli.values():
            stim.autoDraw = False

        # Reset background color
        self.window.color = self.params["BG_COLOR"]

        # Flip the window
        self.window.flip()

    def show_msg(self, text, wait_for_keypress=True):
        """ Show task instructions on screen"""

        # Make a message stimulus
        msg = visual.TextStim(
            self.window,
            text,
            font="Comic Sans MS",
            color=1,
            alignText='center',
            anchorHoriz='center',
            anchorVert='center',
            wrapWidth=self.window.size[0] / 2,
            autoLog=False
        )

        # Change the window color to black
        self.window.color = -1

        # Clear the screen and show the message
        self.clear_screen()
        msg.draw()
        self.window.flip()

        # wait indefinitely, terminates upon any key press
        if wait_for_keypress:
            # Ctrl + C quits the experiment. Resume otherwise.
            pressed = event.waitKeys(modifiers=True)
            if (pressed[0] == 'c') and pressed[1]['ctrl']:
                self.end()
        else:
            core.wait(3)

        # Change the window color back
        self.window.color = self.params["BG_COLOR"]

        # Clear the screen again
        self.clear_screen()

    def log_section(self, name: str, part: str):
        """
        Times and logs the start or end of some section of the experiment.

        Parameters
        ----------
        name : str
            Name of the section. Should be one of 'trial', 'block', 'run'

        part : str
            start or end of a section
        """
        # Run/block/trial params
        _name = name.lower()
        if _name == 'run':
            n = self.run
            sep = '*' * 10
        elif _name == 'block':
            n = self.block
            sep = '=' * 60
        elif _name == 'trial':
            n = self.trial
            sep = '-' * 40
        else:
            raise ValueError("Invalid name.")

        # Clear event buffer
        event.clearEvents()

        # Get the duration and log
        if part == 'start':
            self._log_run(sep)
            self._log_run(f"{name} {n} started: {self.task} task")
        elif part == 'end':
            _mins = int(self.clocks[_name].getTime() // 60)
            _secs = int(self.clocks[_name].getTime() % 60)
            self._log_run(f"{name} {n} ended.")
            self._log_run(f"Duration {_mins} minutes and {_secs} seconds.")
            self._log_run(sep)
        else:
            raise ValueError("Invalid part")

    def end(self, win: bool = True):
        """
        Closes and ends the experiment abruptly.

        Parameters
        ----------
        win : bool
            Whether there is a window to close or not.
        """
        # Log
        logging.info("Quitting the experiment.")

        # Close the window
        if win:
            self.window.close()

        # Quit psychopy and system program
        core.quit()

    def force_quit(self, key_press: str = 'escape'):
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
            logging.warning("Force quitting.")
            self.end()

    def _dir_setup(self) -> Dict:
        """
        Sets up the directories and paths that are important in the experiment. The structure roughly follows the
        BIDS convention.

        Example: data/FIPS/sub-01/staircase/
                 data/FIPS2/sub-02/perceptual/
                 ...
        Returns
        -------
        dict
            data, sub, and task keys and the corresponding Path objects as values.
        """
        # Dictionary for all the directories
        _dirs = dict()

        # data directory
        data_dir = self.HOME / "data" / f"v{self.settings['version']}"
        self.make_dir("data", data_dir)
        _dirs["data"] = data_dir

        # subject directory
        sub_dir = data_dir / f"sub-{self._SUBJECT['id']:02}"
        self.make_dir("subject", sub_dir)
        _dirs["sub"] = sub_dir

        # task directory
        for task in self.TASKS.keys():
            task_dir = sub_dir / task
            self.make_dir(task, task_dir)
            _dirs[task] = task_dir

        # config directory
        _dirs["config"] = self.HOME / "config"

        return _dirs

    def _file_setup(self):
        """
        Sets up files that are going to be saved at the end of the experiment. Naming roughly follows the BIDS
        convention. The files are:
            - Run files: extensions are indicated at the time of saving, but the general run file name is always the
                         same.
                         Example: sub-01_ses-01_run-01_task-saccade_exp-FIPS.<extension>
        Returns
        -------
        dict
            run as keys and the corresponding Path objects as values.
        """
        # Run file name
        run_file = self.paths[self.task] / f"sub-{self._SUBJECT['id']:02d}_ses-{self.session:02d}" \
                                           f"_run-{self.run:02d}_task-{self.task}_exp-{self.NAME}" \
                                           f"_v{self.settings['version']}"
        self.files["run"] = run_file

        # Log filename and data
        log_file = str(run_file) + '.log'
        self.files["run_log"] = log_file
        log_data = logging.LogFile(log_file, filemode='w', level=99)
        self.logs["task"] = log_data

    def _init_logging(self):
        """
        Generates the log file to populate with messages throughout the experiment.

        Returns
        -------
        psychopy.logging.LogFile
        """
        # Make a clock to use for timestamps
        log_clock = core.Clock()
        logging.setDefaultClock(log_clock)  # use that clock for logging

        # Log files
        exp_file = self.paths["sub"] / \
                   f"sub-{self._SUBJECT['id']:02d}_ses-{self.session}_exp-{self.NAME}_v{self.settings['version']}-info.log"
        error_file = self.paths["sub"] / \
                     f"sub-{self._SUBJECT['id']:02d}_ses-{self.session}_exp-{self.NAME}_v{self.settings['version']}-errors.log"

        # Initialize the logfile. Setting the file to the one created with subject's ID.
        exp_log = logging.LogFile(str(exp_file), filemode='a', level=logging.INFO)
        error_log = logging.LogFile(str(error_file), filemode='a', level=logging.ERROR)

        # Control what gets printed to the console
        if self.DEBUG:
            logging.console.setLevel(logging.DEBUG)  # log everything
        else:
            logging.console.setLevel(logging.ERROR)  # log the minimum amount of information

        # Add the tasks levels and placeholder for files
        # logging.addLevel(23, "task")
        # self.logs["task"] = (23, )
        # for t, task in enumerate(self.TASKS.keys()):
        #     log_level = 41 + t
        #     logging.addLevel(log_level, task)
        #     self.logs[task] = [log_level, '']

        # Save the logs
        self.logs["exp"] = exp_log
        self.logs["error"] = error_log
        self.files["exp_log"] = exp_file
        self.files["error_log"] = error_log

        # First log
        logging.exp('************************************************************************************************')
        logging.exp("Experiment started.")
        logging.exp(f"Version: {self.settings['version']}")

    def _log_run(self, log_msg: str):
        """
        Logs a message for a specific run

        Parameters
        ----------
        log_msg : str
            Log message
        """
        logging.log(msg=log_msg, level=99, t=self.clocks["run"].getTime())

    def ms2fr(self, duration: float):
        """ Converts durations from ms to display frames"""
        return np.ceil(duration * self.params["refresh_rate"] / 1000).astype(int)
    
    def recycle_trial(self, trials: np.ndarray, block: np.ndarray, recycle_col: int) -> np.ndarray:
        """
        Takes one trial and adds it to the bottom of a block of trials. It is used for situations where a trial is
        considered failed and needs to be repeated, but not immediately.

        Parameters
        ----------
        trials : np.ndarray
            A block of trials, which is essentially a copy of the "block" parameter below. The difference is that I
            change this copy often, but only use the original block for saving purposes (as done here). This array
            has trials as rows and experiment parameters as columns (features).

        block : np.ndarray
            The original array of experimental block. Only used here for saving whether a specific trial is recycled
            or not.

        recycle_col : int
            Column number for specifying the recycled feature.

        Returns
        -------
        np.ndarray
            A new block of trials, with the first trial removed from the 0 index and added to the bottom of the block
            (-1 index).
        """
        # Log recycling
        self._log_run(f"!!! Recycling trial {self.trial}.")
        self._log_run("-" * 40)

        # Save that the trial is recycled
        block[self.trial - 1, recycle_col] = 1

        # Add the first row to the bottom of the array
        new_trials = np.append(trials, [trials[0]], axis=0)

        # Pop the first burnt row
        new_trials = new_trials[1:]

        return new_trials

    @abstractmethod
    def setup_session(self):
        """
        Creates and prepares visual stimuli that are going to be presented throughout the experiment.
        """
        pass

    @abstractmethod
    def setup_block(self, *args):
        """
        Code to run before each block of the experiment.
        """
        pass

    @abstractmethod
    def setup_trial(self, *args):
        """
        Code to run at the beginning of each trial
        """
        pass

    @abstractmethod
    def bang_trial(self, *args):
        """
        Prepares the block and run, then loops through the blocks and trials.
        Shows the stimuli, monitors gaze, and waits for the saccade.
        Cleans up the trial, block, and experiment, and saves everything.
        """
        pass

    @abstractmethod
    def memorize_block_data(self, *args):
        """
        Saves the data from one block
        """
        pass

    @abstractmethod
    def memorize_session_data(self, **kwargs):
        """
        Saves the session data
        """
        pass
