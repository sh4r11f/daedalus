# -*- coding: utf-8 -*-
# ==================================================================================================== #
#
#
#                    SCRIPT: eyetracking.py
#
#
#          DESCRIPTION: Class for eyetracking experiments
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
from typing import Union, List

import numpy as np
import pandas as pd

import pylink

from daedalus.devices.myelink import MyeLink
from .psyphy import Psychophysics
from .EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy


class Eyetracking(Psychophysics):
    """
    Eyetracking class for running experiments.

    Args:
        project_root (str or Path): The root directory of the project.
        platform (str): The platform where the experiment is running.
        debug (bool): Whether to run the experiment in debug mode or not.
    """
    def __init__(
        self,
        project_root: Union[str, Path],
        platform: str,
        debug: bool,
        tracker_model: str = "Eyelink1000Plus"
    ):

        # Setup
        super().__init__(project_root, platform, debug)

        self.exp_type = "eyetracking"
        self.tracker_model = tracker_model
        self.tracker_params = self.load_config(self.tracker_model)
        self.tracker = self.init_eyelink_tracker()

    def init_eyelink_tracker(self):
        """
        Initializes the eye tracker and connect to it.

        Returns:
            MyeLink: The initialized eye tracker object.
        """
        tracker = MyeLink(self.name, self.tracker_params, self.debug)
        err = tracker.connect()

        if err is None:
            return tracker
        else:
            raise RuntimeError(f"Error in connecting to the tracker: {err}")

    def forge_graphics_env(self):
        """
        Sets up the graphics environment for the calibration.
        """
        # Window
        # Eye tracking experiments always use pixel measurements.
        self.window.units = 'pix'
        # NOTE: If we don't do this early, the eyetracker will override it and we always get a grey background.
        self.window.color = self.exp_params["General"]["background_color"]

        # Configure a graphics environment (genv) for tracker calibration
        genv = EyeLinkCoreGraphicsPsychoPy(self.tracker, self.window)

        # Set background and foreground colors for the calibration target
        foreground_color = self.tracker_params["Calibration"]["target_color"]
        background_color = self.tracker_params["Calibration"]["background_color"]
        genv.setCalibrationColors(foreground_color, background_color)

        # Set up the calibration target
        genv.setTargetType('circle')
        genv.setTargetSize(
            (self.tracker_params["Calibration"]["target_size"], self.tracker_params["Calibration"]["hole_size"])
        )

        # Beeps to play during calibration, validation and drift correction
        # parameters: target, good, error
        #     target -- sound to play when target moves
        #     good -- sound to play on successful operation
        #     error -- sound to play on failure or interruption
        # Each parameter could be ''--default sound, 'off'--no sound, or a wav file
        genv.setCalibrationSounds("", "", "")
        genv.setDriftCorrectSounds("", "", "")

    def run_calibration(self):
        """
        Calibrate the Eyelink 1000
        """
        # A guiding message
        msg = 'Press ENTER and C to recalibrate the tracker.\n\n' \
              'Once the calibration is done, press ENTER and O to resume the experiment.'
        self.show_msg(msg)

        # Initiate calibration
        try:
            self.tracker.doTrackerSetup()
            self._log_run("Tracker calibrated.")
            self.tracker.sendMessage('tracker_calibrated')

        except (RuntimeError, AttributeError) as err:
            self._log_run(f"Tracker not calibrated: {err}")
            self.tracker.exitCalibration()

    def abort_trial(self):
        """Ends recording and recycles the trial"""

        # Stop recording
        if self.tracker.isRecording():
            # add 100 ms to catch final trial events
            pylink.pumpDelay(100)
            self.tracker.stopRecording()

        # clear the screen
        self.clear_screen()

        # Send a message to clear the Data Viewer screen
        bgcolor_RGB = (116, 116, 116)
        self.tracker.sendMessage('!V CLEAR %d %d %d' % bgcolor_RGB)

        # send a message to mark trial end
        # self.tracker.sendMessage(f'TRIAL_RESULT {pylink.TRIAL_ERROR}')
        self.tracker.sendMessage('TRIAL_FAIL')

        # Log
        self._log_run("!!! Trial aborted.")

    def validate_all_gaze(self, gaze_arr: Union[List, np.ndarray], period: str):
        """
        Checks all gaze reports and throws an error if there is any gaze deviation.
        The runtime error should be caught later and handled by recycling the trial.

        Parameters
        ----------
        gaze_arr : list or array
            The array of True or False for each gaze time point.

        period : str
            Name of the period where the validation is being done for.
        """
        # Check all the timepoints
        if not all(gaze_arr):

            # Log
            self._log_run(f"Gaze fail in trial {self.trial}, {period} period.")
            self.tracker.sendMessage(f"{period.upper()}_FAIL")

            # Change fixation color to red
            self.stimuli["fix"].color = [1, -1, -1]
            self.stimuli["fix"].draw()
            self.window.flip()

            # Throw an error
            raise RuntimeError

    def setup_run(self, inst_msg: str):
        """
        Initiates clocks, shows beginning message, and logs start of a run.

        Parameters
        ----------
        inst_msg : str
            A text that has instructions for start of the experiment
        """
        # Setup files for logging and saving this run
        self._file_setup()
        self.files["tracker_local"] = str(self.files["run"]) + '.edf'
        # initiate file on the tracker
        self.open_tracker_file()

        # Clock
        self.clocks["run"].reset()

        # Change the log level task name for this run
        logging.addLevel(99, self.task)

        # Log the start of the run
        self.log_section('Run', 'start')
        self.tracker.sendMessage(f"TASK_START")
        self.tracker.sendMessage(f"TASKID_{self.task}")
        self.tracker.sendMessage(f"RUN_START")
        self.tracker.sendMessage(f"RUNID_{self.exp_run}")

        # Show instruction message before starting
        self.show_msg(inst_msg)

    def setup_block(self, calib=True):
        """
        Sets up an experimental block. Shows a text message and initiates and calibrates the eye tracker.
        """
        # Timing
        self.clocks["block"].reset()

        # Tracker initialization
        self.tracker.setOfflineMode()
        if calib:
            self.run_calibration()

        # clear the host screen
        self.tracker.sendCommand('clear_screen 0')

        # log the beginning of block
        self.log_section('Block', 'start')
        self.tracker.sendMessage(f"BLOCK_START")
        self.tracker.sendMessage(f"BLOCKID_{self.block}")

    def trial_cleanup(self):
        """
        Turns off the stimuli and stop the tracker to clean up the trial.
        """
        # Turn off all stimuli
        self.clear_screen()
        # clear the host screen too
        self.tracker.sendCommand('clear_screen 0')

        # Stop recording frame interval
        self.window.recordFrameIntervals = False

        # Stop recording; add 100 msec to catch final events before stopping
        pylink.pumpDelay(100)
        self.tracker.stopRecording()

    def block_cleanup(self):
        """ Logs the end of a block and shows message about the remaining blocks"""

        # Log the end of the block
        self.log_section("Block", "end")
        self.tracker.sendMessage(f"BLOCKID_{self.block}")
        self.tracker.sendMessage(f"BLOCK_END")

        # Turn everything off
        self.clear_screen()
        # clear the host screen
        self.tracker.sendCommand('clear_screen 0')

        # Get the number of all blocks for this task
        n_blocks = int(self.TASKS[self.task]["blocks"])

        # Show a message between the blocks
        remain_blocks = n_blocks - self.block
        btw_blocks_msg = f"You finished block number {self.block}.\n\n" \
                         f"{remain_blocks} more block(s) remaining in this run.\n\n" \
                         "When you are ready, press the SPACEBAR to continue to calibration."
        self.show_msg(btw_blocks_msg)

    def run_cleanup(self, remain_runs):
        """ Housekeeping at the end of a run"""

        # Log
        self.log_section("Run", "end")
        self.tracker.sendMessage(f"RUNID_{self.exp_run}")
        self.tracker.sendMessage(f"RUN_END")
        self.tracker.sendMessage(f"TASKID_{self.task}")
        self.tracker.sendMessage(f"TASK_END")

        # Turn everything off
        self.clear_screen()
        self.tracker.sendCommand('clear_screen 0')

        # Show an info message
        btw_msg = f"\U00002705 You finished run number {self.run} of the experiment \U00002705\n\n"
        if not self.practice:
            btw_msg += f"There are {remain_runs} run remaining!\n\n"
            btw_msg += "When you are ready, press the SPACEBAR to continue to calibration!"

        end_msg = "\U0000235F\U0000235F\U0000235F \n\n"
        end_msg += "Experiment ended! Thank you for your participation :-)\n\n"
        end_msg += "\U0000235F\U0000235F\U0000235F"

        # Terminate the tracker on the last run
        if remain_runs:
            self.show_msg(btw_msg, wait_for_keypress=False)
            self.terminate_tracker(end=False)
        else:
            self.show_msg(end_msg, wait_for_keypress=False)
            self.terminate_tracker(end=True)

    def save_run_data(self, blocks: list, col_names: list):
        """
        Saves all the blocks and logs to disk.

        Parameters
        ----------
        blocks : list
            All numpy arrays that contain the experiment data

        col_names : list
            Column names for each trial block

        """
        # Compile all blocks into one giant array
        exp_data = np.concatenate(blocks)

        # Make a data frame
        pd_file = str(self.files["run"]) + '.csv'
        df = pd.DataFrame(exp_data, columns=col_names)

        # Save it
        df.to_csv(pd_file, sep=',', index=False)

        # Save the log files
        # logs are in a tuple (or list) with the first item being the log level and the second one the log file
        # self.logs["task"].write(self.files["run_log"])
        # get rid of this log file
        logging.flush()
        logging.root.removeTarget(self.logs["task"])
