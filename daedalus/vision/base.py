#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================================== #
#                                                                                                                                                                                                      #
#                                                                                                                                                                                                      #
#                    SCRIPT: base.py                                                                                                                                                         #
#                                                                                                                                                                                                      #
#                                                                                                                                                                                                      #
#          DESCRIPTION: Basic class for experiments                                                                                                                         #
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
from typing import Dict, Union

from psychopy import core

import logging
from daedalus import log_handler


class Experiment:
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
    def __init__(
            self, 
            name: str, 
            project_root: Union[str, Path], 
            version: float, 
            debug: bool
        ):

        # Setup
        self.exp_type = 'basic'
        self.name = name
        self.project_root = Path(project_root)
        self.version = version
        self.debug = debug

        self.settings = self.read_config('settings')
        self.directories = self._setup_directories()
        self.files = dict()

        # Set up logging
        self.logger = self._init_logging()
        # First log
        self.logger.info(f"Yo! I'm the experiment logger for the day. Let's get started.")
        self.logger.info(f"Our today's experiment is: {self.name}")
        self.logger.info(f"Here are some settings we're working with: ")
        for key, value in self.settings["Study"].items():
            self.logger.info(f"{key}: {value}")

        # Clock to keep track of timing.
        self.clock = core.Clock()

    def read_config(self, conf_name: str):
        """
        Reads and saves parameters specified in a json file under the config directory, e.g. config/params.json

        Args:
            conf_name (str): Name of the file to be loaded.

        Returns:
            dict: The .json file is returned as a python dictionary.
        """
        # Find the parameters file
        params_file = self.project_root / "config" / f"{conf_name}.yaml"

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

    def _setup_directories(self) -> Dict:
        """
        Sets up the directories and paths that are important in the experiment. The structure roughly follows the
        BIDS convention.

        Example: data/FIPS/sub-01/staircase/
                 data/FIPS2/sub-02/perceptual/
                 ...
        Returns
            dict: data, sub, and task keys and the corresponding Path objects as values.
        """
        # Config directory
        config_dir = self.root / "config"

        # Data directory
        data_dir = self.root / "data" / f"v{self.version}"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Log directory
        log_dir = self.root / "log" / f"v{self.version}"
        log_dir.mkdir(parents=True, exist_ok=True)

        dirs = {"data": data_dir, "config": config_dir, "log": log_dir}

        return dirs

    def _init_logging(self):
        """
        Generates the log file for experiment-level logging.
        """
        # Make the logger object 
        logger = log_handler.get_logger(self.name)

        # Log file
        log_file = self.directories["log"] / f"exp_{self.name}_v{self.version}.log"

        # Add new handlers
        log_handler.add_handlers(logger, log_file)

        # Set the level of the handlers
        if self.debug:
            log_handler.set_handlers_level(logger.StreamHandler, logging.DEBUG)

        return logger

    @abstractmethod
    def plan_session(self):
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
