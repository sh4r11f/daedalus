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
from typing import Dict, Union, Tuple

import numpy as np
from scipy import stats

from psychopy import core, monitors, visual, event, info
from psychopy.iohub.devices import Computer

import logging
from daedalus import log_handler, utils


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
    def __init__(self, project_root: Union[str, Path], platform: str, debug: bool):

        # Setup
        self.project_root = Path(project_root)
        self.platform = platform
        self.debug = debug

        self.exp_type = 'basic'
        settings = self.read_config('settings')
        self.study_settings = settings["Study"]
        self.platform_settings = utils.find_in_configs(settings["Platform"], "Name", self.platform)
        self.name = self.study_settings["Name"]
        self.version = self.study_settings["Version"]
        self.directories = self._setup_directories()
        self.files = dict()
        self.runtime_info = dict()

        # Clock to keep track of timing
        self.clock = core.Clock()

        # Set up logging
        self.logger = self._init_logging()

        # Window and monitor
        self.monitor, self.window = self.make_display()

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
        dirs = dict()

        # Project 
        dirs["project"] = self.project_root

        # Config directory
        config_dir = self.root / "config"
        dirs["config"] = config_dir

        # Data directory
        data_dir = self.root / "data" / f"v{self.version}"
        data_dir.mkdir(parents=True, exist_ok=True)
        dirs["data"] = data_dir

        # Log directory
        log_dir = self.root / "log" / f"v{self.version}"
        log_dir.mkdir(parents=True, exist_ok=True)
        dirs["log"] = log_dir

        # Other directories
        modules = self.platform_settings["Modules"]
        for mod in modules:
            name = mod["name"]
            path = mod["path"]
            dirs[name] = path

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

        # First log
        logger.info(f"Greetings, my friend! I'm your experiment logger for the day.")
        logger.info(f"Our today's experiment is: {self.name}.")
        logger.info(f"Here are some settings we're working with: ")
        for key, value in self.study_settings.items():
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
        # Find the monitor name and specification
        monitor_configs = self.load_config('monitors')
        monitor_name = self.platform_settings["Monitor"]
        monitor_specs = monitor_configs[monitor_name]

        # Refresh rate
        rf = monitor_specs["refresh_rate"]
        self.runtime_info["refresh_rate"] = rf

        # Make the monitor object and set width, distance, and resolution
        monitor = monitors.Monitor(
            name=monitor_name,
            width=monitor_specs["size_cm"][0],
            distance=monitor_specs["distance"],
            autoLog=False
        )
        monitor.setSizePix(monitor_specs["size_px"])

        # Gamma correction
        gamma_file = self.directories["config"] / f"{monitor_name}_gamma_grid.npy"
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
            monitor_size_px = monitor_specs["size_px"]
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
        intended_rf = int(self.runtime_info["refresh_rate"])

        if rf is None:
            self.logger.critical("No identical frame refresh times were found.")
            display_warnings += "(✘) No identical frame refresh times were found. You should quit the experiment IMHO.\n\n"
        else:
            # check if the measured refresh rate is the same as the one intended
            rf = np.round(rf).astype(int)
            if rf != intended_rf:
                self.logger.warning(f"The actual refresh rate {rf} does not match the intended refresh rate {intended_rf}.")
                display_warnings += f"(✘) The actual refresh rate {rf} does not match the intended refresh rate {intended_rf}.\n\n"
            else:
                display_warnings += f"(✓) Monitor refresh rate checks out.\n\n"

        # Look for flagged processes
        flagged =run_info['systemUserProcFlagged']
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
                self.end()
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
            self.end()

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