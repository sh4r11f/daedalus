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

from emoji import emojize

import numpy as np
import pandas as pd

from psychopy import core, event, info
from psychopy.tools.monitorunittools import deg2pix, pix2deg
import psychopy.gui.qtgui as gui
from psychopy.iohub.devices import Computer

from daedalus.factory import (
    FileManager, SettingsManager, DataManager, TimeManager,
    StimulusFactory, DisplayFactory, TaskFactory
)
from daedalus.codes import Codex
from daedalus import log_handler, utils


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
        name (str): Name of the experiment.
        version (float): Version of the experiment.
        root (str or Path): Home directory of the experiment.
        platform (str): Platform for the experiment.
        debug (bool): Debug mode for the experiment.
    """
    def __init__(self, name, version, root, task_name, platform, mode):

        # Setup
        self.exp_type = 'psychophysics'
        self.name = name
        self.version = version
        self.root = Path(root)
        self.task_name = task_name
        self.platform = platform
        if mode == "debug":
            self.debug = True
            self.simulation = False
        elif mode == "simulation":
            self.debug = True
            self.simulation = True
        else:
            self.debug = False
            self.simulation = False

        # Directories and files
        self.files = FileManager(name, root, version, self.debug)

        # Settings
        self.settings = SettingsManager(self.files.dirs.config, version, platform)

        # Log
        self.codex = Codex()
        self.logger = None

        # Visuals
        self.display = DisplayFactory(
            monitor_name=self.settings.platform["Monitor"],
            monitor_params=self.settings.monitor,
            gamma=self.settings.gamma,
            window_params=self.settings.stimuli["Display"]
        )
        self.stimuli = StimulusFactory(
            root,
            self.settings.stimuli,
            self.settings.platform["Directories"]["fonts"]
        )

        # Subject, session, block, and trial
        self.sub_id = None
        self.ses_id = None
        self.settings.select_task(self.task_name)
        self.task = TaskFactory(self.task_name, self.settings.task)
        self.block = None
        self.trial = None

        # Data
        self.data = DataManager(self.name, self.settings.exp, self.version)

        # Clocks
        self.timer = TimeManager()

    def init_session(self):
        """
        Starts up the experiment by initializing the settings, directories, files, logging, monitor, window, and clocks.
        """
        self.timer.start()
        self.data.load_participants(self.files.participants)

        # Ask for subject information
        select, expert = self.hello_gui()
        if select == "Load":
            sub_info = self.subject_loadation_gui()
        else:
            sub_info = self.subject_registration_gui()

        # Session and task info
        ses, _ = self.session_task_selection_gui(sub_info)

        # Save the information
        sub_info["Experiment"] = self.name
        sub_info["Task"] = self.task_name
        sub_info["Version"] = self.version
        sub_info["Session"] = ses
        sub_info["Experimenter"] = expert
        self.sub_id = self._fix_id(sub_info["PID"].values[0])
        self.ses_id = self._fix_id(ses)

        # Initialize files
        self.files.add_session(self.sub_id, self.ses_id)

        # Initialize the logging
        self.logger = log_handler.DaedalusLogger(self.name, self.debug, self.files.log)

        # Make visuals
        self.display.start(self.debug)
        self.stimuli.set_display(self.display.monitor, self.display.window)
        self.stimuli.make_message("exp_msg")

        # Save the subject info
        self.data.sub_id = self.sub_id
        self.data.ses_id = self.ses_id
        self.data.task_name = self.task_name
        self.data.add_participant(sub_info)

        # Load blocks
        self.task.load_blocks()

        # Save subject and log the start of the session
        self.logger.info("Greetings, my friend! I'm your experiment logger for the day.")
        self.logger.info(f"Looks like we're running a {self.exp_type} experiment called {self.name} (v{self.version}).")
        self.logger.info(f"Today is {self.timer.today}.")
        self.logger.info(f"Subject {self.sub_id}")
        self.logger.info(f"Session {self.ses_id}")
        self.logger.info(f"Task: {self.task_name}.")
        self.logger.info("Let's get started!")
        self.logger.info("-" * 80)

    def prepare_block(self, block):
        """
        Prepare the block for the task.
        """
        self.block_id = self._fix_id(block.id)

        # Show the start of the block message
        if block.repeat:
            txt = f"You are repeating block {self.block_id}/{self.task.n_blocks:02d}.\n\n"
        else:
            txt = f"You are about to begin block {self.block_id}/{self.task.n_blocks:02d}.\n\n"
        txt += "Press Space to start."
        while True:
            resp = self.show_msg(txt)
            if (resp == "space") or (self.debug):
                # Reset the block clock
                self.timer.start_block()
                # Initialize the block data
                errors = self.files.add_block()
                if errors:
                    for e in errors:
                        self.block_warning(e)
                self.data.init_stimuli()
                self.data.init_behavior()
                self.data.init_frames()
                # Log the start of the block
                self.block_info(self.codex.message("block", "init"))
                self.block_info(f"BLOCKID_{self.block_id}")
                break

    def prepare_trial(self, trial):
        """
        Prepare the trial for the block.
        """
        self.trial_id = self._fix_id(trial.id)
        self.timer.start_trial()
        self.trial_info(self.codex.message("trial", "init"))
        self.trial_info(f"TRIALID_{self.trial_id}")

    def wrap_trial(self):
        """
        Wrap up the trial.
        """
        self.trial_info(self.codex.message("trial", "fin"))
        self.display.window.recordFrameIntervals = False
        self.display.window.frameIntervals = []
        self.display.clear()

    def stop_trial(self, trial):
        """
        Stop the trial.
        """
        self.trial_warning(self.codex.message("trial", "stop"))
        self.trial_warning(trial.error)
        self.display.window.flip()
        self.display.window.recordFrameIntervals = False
        self.display.window.frameIntervals = []
        self.display.clear()

    def wrap_block(self, block):
        """
        Wrap up the block.
        """
        # Log the end of the block
        self.block_info(self.codex.message("block", "fin"))

        # Save the data
        self.data.save_behavior(self.files.behavior)
        self.data.save_stimuli(self.files.stim_data)
        self.data.save_frames(self.files.frames)

        # Show the end of the block message
        if block.name == "practice":
            txt = "Practice block is over."
        else:
            txt = f"Block {self.block_id} is finished."
        txt += "\n\nStay tuned..."
        self.show_msg(txt, wait_time=self.settings.stimuli["Message"].get("wait_duration", 3))
        self.display.clear()

    def stop_block(self, block):
        """
        Stop the block.
        """
        self.block_error(self.codex.message("block", "stop"))
        self.block_error(block.error)
        msg = f"An error has occured: {block.error}\n\nWe have to stop this block..."
        self.show_msg(msg, wait_time=self.settings.stimuli["Message"]["warning_duration"], msg_type="warning")

    def turn_off(self):
        """
        End the session.
        """
        # Log the end of the session
        self.logger.info(self.codex.message("ses", "fin"))
        msg = f"You did it. Session {self.ses_id} of the experiment is over."
        msg += "\n\nThank you!"
        self.show_msg(msg, wait_time=self.settings.stimuli["Message"]["duration"], msg_type="info")
        # Save
        self.data.participants["Completed"] = pd.to_datetime(self.data.participants["Completed"])
        self.data.update_participant("Completed", self.timer.today)
        self.data.save_participants(self.files.participants)
        # Quit
        self.goodbye()

    def block_debug(self, *args):
        """
        Log a debug message for the block.

        Args:
            msg (str): The message to be logged.
        """
        if len(args) == 1:
            self.logger.debug(args[0], extra={"block": self.block_id}, stack_info=True, stacklevel=4)
        else:
            msg = " ".join([str(a) for a in args])
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

    def hello_gui(self):
        """
        GUI for choosing/registering subjects.

        Returns:
            str: The choice made by the user.

        Raises:
            ValueError: If the experiment is cancelled.
        """
        emos = self.settings.stimuli["Emojis"]
        dlg = gui.Dlg(
            title="Yo!",
            font=self.settings.stimuli["Display"]["gui_font"],
            font_size=self.settings.stimuli["Display"]["gui_font_size"],
            size=int(self.settings.stimuli["Display"]["gui_size"]),
            labelButtonOK=emojize(f"Let's Go {emos['lets_go']}"),
            labelButtonCancel=emojize(f"Go Away {emos['go_away']}"),
            alwaysOnTop=True
        )
        # Add the experiment info
        color = utils.str2tuple(self.settings.stimuli["Display"]["gui_text_color"])
        dlg.addText("Experiment", color=color)
        dlg.addFixedField(key="title", label=emojize(f"{emos['title']}Title"), initial=emojize(f"{self.name}"))
        dlg.addFixedField(key="date", label=emojize(f"{emos['date']}Date"), initial=self.timer.today)
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
            choices=[emojize(a["name"]) for a in self.settings.study["Authors"]]
        )
        dlg.addField(
            key="selection",
            label=emojize(f"{emos['participant']} Participant"),
            choices=[emojize("Register"), emojize("Load")]
        )

        if self.simulation:
            return self.settings.simulation["selection"], self.settings.simulation["experimenter"]
        else:
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
        emos = self.settings.stimuli["Emojis"]
        dlg = gui.Dlg(
            title=f"{self.name} Experiment",
            font=self.settings.stimuli["Display"]["gui_font"],
            font_size=self.settings.stimuli["Display"]["gui_font_size"],
            size=self.settings.stimuli["Display"]["gui_size"],
            labelButtonOK=emojize(f"Register {emos['register']}"),
            labelButtonCancel=emojize(f"Go Away {emos['go_away']}"),
            alwaysOnTop=True
        )

        # Check what PIDs are already used
        color = utils.str2tuple(self.settings.stimuli["Display"]["gui_text_color"])
        dlg.addText(
            emojize(f"{emos['participant']} Participant"),
            color=color
        )
        min_pid, max_pid = self.settings.exp["Subjects"]["PID"].split("-")
        pids = [f"{i:02d}" for i in range(int(min_pid), int(max_pid) + 1)]
        burnt_pids = self._fix_id(self.data.participants["PID"].values)
        available_pids = []
        for pid in pids:
            if pid not in burnt_pids:
                available_pids.append(pid)

        # Add fields
        dlg.addField(key="PID", label=emojize(f"{emos['pid']} PID"), choices=available_pids)
        required_info = self.settings.exp["Subjects"]
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
        if self.simulation:
            info = dlg.data
            for col in self.data.participants.columns:
                if col not in info.keys():
                    info[col] = None
            info["PID"] = self.settings.simulation["pid"]
            return pd.DataFrame.from_dict({k: [v] for k, v in info.items()}, orient="columns")
        else:
            info = dlg.show()
            if dlg.OK:
                for col in self.data.participants.columns:
                    if col not in info.keys():
                        info[col] = None
                return pd.DataFrame.from_dict({k: [v] for k, v in info.items()}, orient="columns")
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
        emos = self.settings.stimuli["Emojis"]
        dlg = gui.Dlg(
            title="Participant Loadation",
            font=self.settings.stimuli["Display"]["gui_font"],
            font_size=self.settings.stimuli["Display"]["gui_font_size"],
            size=self.settings.stimuli["Display"]["gui_size"],
            labelButtonOK=emojize(f"Load {emos['load']}"),
            labelButtonCancel=emojize(f"Go Away {emos['go_away']}"),
            alwaysOnTop=True
        )

        # Find the PID and initials that are registered
        color = utils.str2tuple(self.settings.stimuli["Display"]["gui_text_color"])
        dlg.addText(
            emojize(f"{emos['participant']} Participant"),
            color=color
        )
        pids = self._fix_id(self.data.participants["PID"].values)
        initials = self.data.participants["Initials"].values
        pid_initials = [f"{pid} ({initial})" for pid, initial in zip(pids, initials)]
        dlg.addField(key="sub", label=emojize(f"{emos['pid']} PID (Initials)"), choices=pid_initials)

        # Show the dialog
        if self.simulation:
            info = dlg.data
            pid = self.settings.simulation["pid"]
            return self.data.participants.loc[self.data.participants["PID"].astype(int) == int(pid), :]
        else:
            info = dlg.show()
            if dlg.OK:
                # Load the subject info
                pid = info["sub"].split(" ")[0]
                sub_df = self.data.participants.loc[self.data.participants["PID"].astype(int) == int(pid), :]
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
        emos = self.settings.stimuli["Emojis"]
        dlg = gui.Dlg(
            title=f"{self.name} Experiment",
            font=self.settings.stimuli["Display"]["gui_font"],
            font_size=self.settings.stimuli["Display"]["gui_font_size"],
            size=self.settings.stimuli["Display"]["gui_size"],
            labelButtonOK=emojize(f"Alright {emos['alright']}"),
            labelButtonCancel=emojize(f"Go Away {emos['go_away']}"),
            alwaysOnTop=True
        )
        # Add the subject info
        color = utils.str2tuple(self.settings.stimuli["Display"]["gui_text_color"])
        dlg.addText(
            emojize(f"{emos['participant']} Participant Information"),
            color=color
        )
        for field in sub_df.columns:
            for valid in self.settings.exp["Subjects"].keys():
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
        tasks = list(self.settings.exp["Tasks"].keys())
        done = sub_df.loc[~(pd.isnull(sub_df["Completed"])), ["Session", "Task"]]
        remaining = [task for task in tasks if task not in done["Task"].values]
        if self.task_name not in remaining:
            dlg.addFixedField(key="error", label="ERROR!", initial="This task is not available for this participant.")
        else:
            choices = []
            for ses in self.settings.exp["Sessions"]:
                if ses["id"] not in done["Session"].values:
                    choices.append(f"{ses['id']}")
        # for ses in self.settings.exp["Sessions"]:
        #     if ses["id"] not in complete["Session"].values:
        #         for task in ses["tasks"]:
        #             choices.append(f"{ses['id']} ({task})")
        #     else:
        #         burnt_tasks = complete.loc[complete["Session"] == ses["id"], "Task"].values
        #         available_tasks = [task for task in ses["tasks"] if task not in burnt_tasks]
        #         for task in available_tasks:
        #             choices.append(f"{ses['id']} ({task})")
        dlg.addField(
            key="choice",
            label=emojize(f"{emos['session']} Session (Task)"),
            choices=choices,
        )

        # Show the dialog
        if self.simulation:
            info = dlg.data
            return self.settings.simulation["session"], self.task_name
        else:
            info = dlg.show()
            if dlg.OK:
                # ses, task = info["choice"].split(" ")
                ses = info["choice"]
                ses = self._fix_id(ses)
                # task = task[1:-1]
                task = self.task_name
                return ses, task
            else:
                raise SystemExit("Session selection cancelled.")

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
            version=self.version,
            win=self.display.window,
            refreshTest='grating',
            userProcsDetailed=True,
            verbose=True
        )

        # Start testing
        results = []
        self.logger.info("Running system checks.")
        results.append("<Press Space to accept or Escape to quit>\n\n")

        # Refresh rate
        rf = self.display.window.getActualFrameRate(
            nIdentical=20,
            nMaxFrames=100,
            nWarmUpFrames=10,
            threshold=rf_thresh,
            infoMsg="||  Warming Up  ||"
        )
        intended_rf = int(self.settings.monitor["refresh_rate"])
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
            self.logger.warning("Flagged processes:")
            w = ""
            for proc in np.unique(flagged):
                w += f"\t- {proc}\n"
                self.logger.warning(f"\t- {proc}")
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
        if self.settings.platform["OS"] in ["darwin", "Mac OS X"]:
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
        # Get variables
        event.clearEvents()
        params = self.settings.stimuli["Message"]
        stim = self.stimuli.exp_msg
        win = self.display.window

        # Change background color
        win.color = utils.str2tuple(params["background_color"])
        win.flip()

        # Set the text
        stim.text = text

        # Change color based on info
        if msg_type == "info":
            stim.color = utils.str2tuple(params["normal_text_color"])
        elif msg_type in ["warning", "error"]:
            stim.color = utils.str2tuple(params["warning_text_color"])
        elif msg_type == "good_news":
            stim.color = utils.str2tuple(params["good_news_text_color"])

        # Show the message
        stim.draw()
        win.flip()

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

        if self.simulation:
            key_press = self.settings.simulation["next_key"]
            core.wait(self.settings.simulation["wait_text"])
        else:
            key_press = None
            while True:
                # Check for key presses
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

                if wait_keys:
                    if key_press is not None:
                        break
                # Wait for a certain amount of time
                if wait_time:
                    core.wait(wait_time)
                    break

        if key_press == "ctrl+c":
            self.goodbye(self.codex.message("usr", "term"))
        else:
            self.display.clear()
            return key_press

    def goodbye(self, raise_error=None):
        """
        Closes and ends the experiment.
        """
        # Close the window
        self.display.clear()
        self.display.close()

        # Quit
        if raise_error is not None:
            self.logger.critical(self.codex.message("exp", "term"))
            try:
                # Save as much as you can
                self.data.save_participants(self.files.participants)
                self.data.save_stimuli(self.files.stim_data)
                self.data.save_behavior(self.file.behavior)
                self.data.save_frames(self.files.frames)
                self.logger.close_file()
            except Exception as e:
                print(f"Error in saving data: {e}")
                raise SystemExit(f"Experiment ended with error: {raise_error}")
        else:
            # Log
            msg = f"Bye Bye Experiment. We only knew each other for {self.timer.exp.getTime() / 60:.2f} minutes."
            self.logger.info(msg)
            self.logger.close_file()
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
        """
        Converts durations from ms to display frames:
            duration (ms) * 1/1000 (s/ms) * rf (frames/s) = n (frames)

        Args:
            duration (float): Duration in ms.

        Returns:
            int: Duration in frames
        """
        return np.ceil(duration * (1/1000) * self.settings.monitor["refresh_rate"]).astype(int)

    def fr2ms(self, n: int):
        """
        Converts durations from display frames to ms:
            n (frames) * 1/rf (s/frame) * 1000 (ms/s) = duration (ms)

        Args:
            n (int): Duration in frames.

        Returns:
            float: Duration in ms.
        """
        return n * (1/self.settings.monitor["refresh_rate"]) * 1000

    def hz2cpf(self, cps: float):
        """
        Converts cycles per second to cycles per frame:
            cps (cycles/s) * 1/rf (s/frames) = cpf (cycles/frame)

        Args:
            cps (float): Cycles per second (temporal frequency).

        Returns:
            float: Cycles per frame
        """
        return cps * (1/self.settings.monitor["refresh_rate"])

    def cpf2hz(self, cpf: float):
        """
        Converts cycles per frame to cycles per second:
            cpf (cycles/frame) * rf (frame/s) = cps (cycles/s)

        Args:
            cpf (float): Cycles per frame.

        Returns:
            float: Cycles per second.
        """
        return cpf * self.settings.monitor["refresh_rate"]

    def cpf2dps(self, cpf: float, cpd: float):
        """
        Converts cycles per frame to degrees per second:
            cpf (cycles/frame) * 1/cpd (dva/cycles) * rf (frames/s) = dps (dva/s)

        Args:
            cpf (float): Cycles per frame.
            cpd (float): Cycles per degree (spatial frequency).

        Returns:
            float: Degrees per second.
        """
        return cpf * (1/cpd) * self.settings.monitor["refresh_rate"]

    def dps2pps(self, dps: float):
        """
        Converts degrees per second to pixels per second:
            dps (dva/s) * ppd (pix/dva) = pps (pix/s)

        Args:
            dps (float): Degrees per second.

        Returns:
            float: Pixels per second.
        """
        ppd = deg2pix(1, self.display.monitor)
        return dps * ppd

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
        return cps * (1/cpd) * (1/self.settings.monitor["refresh_rate"])

    def hz2dps(self, cps: float, cpd: float):
        """
        Converts cycles per second to degrees per second:
            cps (cycles/s) * 1/cpd (dva/cycles) = dps (dva/s)

            (v = lambda * f)

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
        ppd = deg2pix(1, self.display.monitor)
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
        ppd = deg2pix(1, self.display.monitor)
        return cps * (1/cpd) * ppd * (1/self.settings.monitor["refresh_rate"])

    def cpd2cpp(self, cpd: float):
        """
        Converts cycles per degree to cycles per pixel:
            cpd (cycles/dva) * dpp (dva/pixel) = cpp (cycles/pixel)

        Args:
            cpd (float): Cycles per degree (spatial frequency).

        Returns:
            float: Cycles per pixel.
        """
        dpp = pix2deg(1, self.display.monitor)
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
        ppd = deg2pix(1, self.display.monitor)
        return cpp * ppd

    def pps2dps(self, pps: float, ppd=None):
        """
        Converts pixels per second to degrees per second:
            pps (pix/s) * 1/ppd (dva/pix) = dps (dva/s)

        Args:
            pps (float): Pixels per second.

        Returns:
            float: Degrees per second.
        """
        if ppd is None:
            ppd = pix2deg(1, self.display.monitor)
        return pps * (1/ppd)

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
        return np.round(val, self.settings.exp["General"]["round_decimal"])

    def ms_round(self, times):
        return np.round(times * 1000, self.settings.exp["General"]["round_decimal"])
