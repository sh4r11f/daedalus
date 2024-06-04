#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================================== #
#                                                                                                                                                                                                      #
#                                                                                                                                                                                                      #
#                    SCRIPT: psyphy.py                                                                                                                                                     #
#                                                                                                                                                                                                      #
#                                                                                                                                                                                                      #
#          DESCRIPTION: Class for psychophysics experiments                                                                                                          #
#                                                                                                                                                                                                      #
#                                                                                                                                                                                                      #
#                       RULE: DAYW                                                                                                                                                            #
#                                                                                                                                                                                                      #
#                                                                                                                                                                                                      #
#                                                                                                                                                                                                      #
#                  CREATOR: Sharif Saleki                                                                                                                                                #
#                         TIME: 05-26-2024-7810598105114117                                                                                                                #
#                       SPACE: Dartmouth College, Hanover, NH                                                                                                               #
#                                                                                                                                                                                                      #
# ==================================================================================================== #
from pathlib import Path

from abc import abstractmethod
from typing import Union

import pandas as pd

from psychopy import gui, data

from .base import Experiment


class Psychophysics(Experiment):
    """
    Base class for all the experiments. It contains necessary methods and attributes.

    Args:
        project_root (str): The root directory of the project.
        platform (str): The platform to run the experiment on.
        debug (bool): Whether to run the experiment in debug mode.

    Attributes:
        exp_type (str): The type of the experiment.
        exp_params (dict): The parameters of the experiment.
        stim_params (dict): The parameters of the stimuli.
        subject_info (dict): The information about the subject.
        files (dict): The files used in the experiment.
        directories (dict): The directories used in the experiment.
    """
    def __init__(self, project_root: Union[str, Path], platform: str, debug: bool):

        # Setup
        super().__init__(project_root, platform, debug)
        self.exp_type = 'psychophysics'

        # Configuration
        self.exp_params = self.load_config('experiment')
        self.stim_params = self.load_config('stimuli')

        # Data
        self.subject_info = dict()

    def choose_session_gui(self):
        """
        GUI for choosing the session.
        """
        n_sessions = self.exp_params["Counts"]["sessions_per_subject"]
        dlg = gui.Dlg(title="Choose Session", labelButtonOK="Select", labelButtonCancel="Exit")
        dlg.addText("Experiment", self.name, color="blue")
        dlg.addText("Date", data.getDateStr(), color="blue")
        dlg.addFixedField("Version", self.version)
        dlg.addField("Session", choices=[str(i) for i in range(1, n_sessions + 1)])

        dlg.show()
        if dlg.OK:
            return dlg.data
        else:
            return None

    def register_subject_gui(self, burnt_PIDs: list = None):
        """
        GUI for registering the subject.
        """
        dlg = gui.Dlg(title="Register Subject", labelButtonOK="Register", labelButtonCancel="Exit")
        dlg.addText("Experiment", self.name, color="blue")
        dlg.addText("Date", data.getDateStr(), color="blue")
        dlg.addFixedField("Version", self.version)

        required_info = self.exp_params["Info"]["Subjects"]
        valid_pids = [f"{id:02d}" for id in range(1, int(self.exp_params["Counts"]["subjects"]) + 1)]
        if burnt_PIDs is not None:
            valid_pids = [pid for pid in valid_pids if pid not in burnt_PIDs]

        for field in required_info:
            if isinstance(required_info[field], list):
                dlg.addField(field, choices=required_info[field])
            else:
                if field == "PID":
                    dlg.addField(field, choices=valid_pids)
                else:
                    dlg.addField(field)

        dlg.show()
        if dlg.OK:
            return dlg.data
        else:
            return None

    def load_subject_gui(self, valid_PIDs: list):
        """
        GUI for loading the subject.
        """
        # Make the dialog
        dlg = gui.Dlg(title="Load Subject", labelButtonOK="Load", labelButtonCancel="Exit")
        dlg.addText("Experiment", self.name, color="blue")
        dlg.addText("Date", data.getDateStr(), color="blue")
        dlg.addFixedField("Version", self.version)
        dlg.addField("PID", choices=valid_PIDs)

        # Show the dialog
        dlg.show()

        if dlg.OK:

            # Load the subject info
            subject_pid = dlg.data[0]
            subject_info = self.load_subject_info(subject_pid)

            # Display the subject info to get confirmation
            conf_dlg = gui.Dlg(title=f"Found info for participant {subject_pid}", labelButtonOK="Confirm", labelButtonCancel="Exit")
            conf_dlg.addText("Please confirm the following information.", self.name, color="red")
            for field, info in subject_info:
                conf_dlg.addFixedField(field, info)

            # Show the dialog
            conf_dlg.show()
            if conf_dlg.OK:
                return subject_info
            else:
                return None
        else:
            return None

    def load_subject_info(self, pid: str):
        """
        Loads the subject info from the subject file.
        """
        df = pd.read_csv(self.files["participants"], sep="\t")
        if pid not in df["PID"].values:
            raise ValueError(f"PID {pid} not found in the participant file.")
        else:
            subject_info = df[df["PID"] == pid].to_dict(orient="records")[0]
            return subject_info

    def init_participants_file(self):
        """
        Initializes the participants file.
        """
        participant_file = self.directories["data"] / "participants.tsv"
        self.files["participants"] = participant_file

        if not participant_file.exists():
            columns = list(self.exp_params["Info"]["Subjects"].keys()) + ["Date", "Experiment", "Version"]
            df = pd.DataFrame(columns=columns)
            df.to_csv(participant_file, sep="\t", index=False)
        else:
            return None

    def save_subject_info(self, subject_info: dict):
        """
        Saves the subject info to the participants file.
        """
        df = pd.read_csv(self.files["participants"], sep="\t")
        sub_df = pd.DataFrame([subject_info])
        out_df = pd.concat([df, sub_df], ignore_index=True)
        out_df.to_csv(self.files["participants"], sep="\t", index=False)

    @abstractmethod
    def plan_session(self, **kwargs):
        """
        Creates and prepares visual stimuli that are going to be presented throughout the experiment.
        """
        pass

    @abstractmethod
    def plan_block(self, *args):
        """
        Code to run before each block of the experiment.
        """
        pass

    @abstractmethod
    def plan_trial(self, *args):
        """
        Code to run at the beginning of each trial
        """
        pass

    @abstractmethod
    def do_session(self, *args):
        """
        Prepares the session, then loops through the blocks and trials.
        Shows the stimuli, monitors gaze, and waits for the saccade.
        Cleans up the trial, block, and experiment, and saves everything.
        """
        pass

    @abstractmethod
    def do_block(self, *args):
        """
        Prepares the block and run, then loops through the blocks and trials.
        Shows the stimuli, monitors gaze, and waits for the saccade.
        Cleans up the trial, block, and experiment, and saves everything.
        """
        pass

    @abstractmethod
    def do_trial(self, *args):
        """
        Prepares the block and run, then loops through the blocks and trials.
        Shows the stimuli, monitors gaze, and waits for the saccade.
        Cleans up the trial, block, and experiment, and saves everything.
        """
        pass
    
    @abstractmethod
    def memorize_session(self, **kwargs):
        """
        Saves the session data
        """
        pass

    @abstractmethod
    def memorize_block(self, *args):
        """
        Saves the data from one block
        """
        pass

    @abstractmethod
    def memorize_trial(self, *args):
        """
        Saves the data from one block
        """
        pass

    @abstractmethod
    def memorize_subject(self, *args):
        """
        Saves the data from one block
        """
        pass
