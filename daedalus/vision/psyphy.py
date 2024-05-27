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
import yaml
from pathlib import Path

from abc import abstractmethod
from typing import Union, Tuple

import numpy as np
from scipy import stats

from psychopy import gui, data
from psychopy.iohub.devices import Computer

from . import Experiment


class PsychophysicsExperiment(Experiment):
    """
    Base class for all the experiments. It contains necessary methods and attributes.
    """
    def __init__(self, project_root: Union[str, Path], platform: str, debug: bool):

        # Setup
        super().__init__(project_root, platform, debug)
        
        self.exp_type = 'psychophysics'
        self.parameters = self.load_config('parameters')
        self.exp_params = self.find_in_configs(self.parameters["Experiment"], "Version", self.version)
        self.stim_params = self.find_in_configs(self.parameters["Stimuli"], "Version", self.version)
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
            return dlg.data[0]
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
        for field in required_info:
            if isinstance(required_info[field], list):
                dlg.addField(field, choices=required_info[field])
            else:

                dlg.addField(field)

        dlg.show()
        if dlg.OK:
            for i, field in enumerate(required_info):
                self.subject_info[field] = dlg.data[i]
            return True
        else:
            return False
        
    def load_subject_gui(self):
        """
        GUI for loading the subject.
        """
        dlg = gui.Dlg(title="Load Subject", labelButtonOK="Load", labelButtonCancel="Exit")
        dlg.addText("Experiment", self.name, color="blue")
        dlg.addText("Date", data.getDateStr(), color="blue")
        dlg.addFixedField("Version", self.version)
        dlg.addField("Subject ID")

        dlg.show()
        if dlg.OK:
            return dlg.data[0]
        else:
            return None

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
