# -*- coding: utf-8 -*-
# ==================================================================================================== #
#                                                                                                                                                                                                      #
#                                                                                                                                                                                                      #
#                    SCRIPT: eyetracking.py                                                                                                                                              #
#                                                                                                                                                                                                      #
#                                                                                                                                                                                                      #
#          DESCRIPTION: Class for eyetracking experiments                                                                                                               #
#                                                                                                                                                                                                      #
#                                                                                                                                                                                                      #
#                       RULE: DAYW                                                                                                                                                            #
#                                                                                                                                                                                                      #
#                                                                                                                                                                                                      #
#                                                                                                                                                                                                      #
#                  CREATOR: Sharif Saleki                                                                                                                                                #
#                         TIME: 05-26-2024-[78 105 98 105114117]                                                                                                           #
#                       SPACE: Dartmouth College, Hanover, NH                                                                                                               #
#                                                                                                                                                                                                      #
# ==================================================================================================== #
from pathlib import Path
from typing import Union, List, Tuple, Dict

import numpy as np
import pandas as pd

import pylink
from psychopy.tools.monitorunittools import deg2pix

from daedalus.devices.myelink import MyeLink
from .psyphy import Psychophysics
from .EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy


class Eyetracking(Psychophysics):
    """
    """
    def __init__(self, project_root: Union[str, Path], platform: str, debug: bool):

        # Setup
        super().__init__(project_root, platform, debug)
        self.exp_type = "eyetracking"
        self.tracker_model = "Eyelink1000Plus"

        # Parameters
        self.tracker_params = self.load_config("eyetracker")[self.tracker_model]

        # Window
        # Eye tracking experiments always use pixel measurements.
        self.window.units = 'pix'
        # NOTE: If we don't do this early, the eyetracker will override it and we always get a grey background.
        self.window.color = self.exp_params["General"]["background_color"]

        # Eyelink-specific .edf files
        self.files["tracker_host"] = f"{self.settings["Study"]["Shorthand"]}{self._SUBJECT['id']:02d}.edf"

        # Initialize the eye tracker
        self.tracker = MyeLink(self.tracker_params, self.name, self.window, self.debug)

    def cook_calibration(self):
        """
        Calibrates the Eyelink 1000
        """
        # Connect to the tracker
        self.hook_tracker()

        # Configure the tracker
        self.tracker.confiugre()

        # Setup calibration
        self.concot_calibration()

        # Open the EDF file
        self.open_tracker_file()

        # Start calibration
        calibrated = self.tracker.calibrate()
        if not calibrated:
            self.logger.critical(f'Error in calibrating the tracker: {self.tracker.error}')
            self.goodbye()

    def concot_calibration(self):
        """
		Calibration window for the Eyelink 1000
		"""
  		# Configure a graphics environment (genv) for tracker calibration
        genv = EyeLinkCoreGraphicsPsychoPy(self.tracker, self.window)

		# Set background and foreground colors for the calibration target
        foreground_color = self.tracker_params["Calibration"]["target_color"]
        background_color = self.tracker_params["Calibration"]["background_color"]
        genv.setCalibrationColors(foreground_color, background_color)

		# Set up the calibration target
        genv.setTargetType('circle')

		# Configure the size of the calibration target (in pixels)
		# this option applies only to "circle" and "spiral" targets
        target_size = deg2pix(float(self.tracker_params["Calibration"]["target_size"]), self.monitor)
        genv.setTargetSize(target_size)

		# Beeps to play during calibration, validation and drift correction
		# parameters: target, good, error
		#     target -- sound to play when target moves
		#     good -- sound to play on successful operation
		#     error -- sound to play on failure or interruption
		# Each parameter could be ''--default sound, 'off'--no sound, or a wav file
        genv.setCalibrationSounds('', '', '')

		# Request Pylink to use the PsychoPy window we opened above for calibration
        pylink.openGraphicsEx(genv)
  
    def hook_tracker(self):
        """
        Connects to an Eyelink 1000 using its python API (pylink).
        """
        connection = self.tracker.connect()
        if not connection:
            self.logger.critical(f'Error in connecting to the tracker: {self.tracker.error}')
            self.goodbye()

    def open_tracker_file(self):
        """
        Creates an edf file on the host Eyelink computer to record the data
        """
        opened = self.tracker.open_file(self.files["tracker_host"])
        if not opened:
            self.logger.critical(f'Error in opening the host file: {self.tracker.error}')
            self.goodbye()

    def run_calibration(self):
        """Calibrate the Eyelink 1000"""

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

    def start_recording(self):
        """Initiate recording of eye data by the tracker"""

        try:
            # arguments: sample_to_file, events_to_file, sample_over_link,
            # event_over_link (1-yes, 0-no)
            self.tracker.startRecording(1, 1, 1, 1)
            self._log_run("Tracker recording.")
            self.tracker.sendMessage("recording")

        except RuntimeError as err:
            self._log_run(f"Tracker not recording: {err}")
            self.tracker.sendMessage("not_recording")
            return pylink.TRIAL_ERROR

    def check_connection(self):
        """Checks if the tracker is still alive"""

        # Get the status
        error = self.tracker.isConnected()  # returns 1 if connected, 0 if not connected, -1 if simulated

        # Check it only in the actual running mode because simulated tracker returns -1 in isConnected()
        if not self.DUMMY:

            # For some reason TRIAL_OK is 0 in pylink
            if error is pylink.TRIAL_OK:
                # Log the disconnect
                self.tracker.sendMessage('tracker_disconnected')
                logging.error(f"Tracker disconnected. Error: {error}")
                self.end()

    def drift_correction(self, fix_pos):
        """
        Performs drift correction with the Eyelink 1000

        Parameters
        ----------
        fix_pos : list or tuple
            Coordinates of the fixation spot
        """
        # unpack the fixation
        fix_x, fix_y = fix_pos

        # Apply the drift correction
        self.tracker.sendCommand('driftcorrect_cr_disable = OFF')
        self.tracker.sendCommand(f'online_dcorr_refposn {int(self.window.size[0] / 2)},{int(self.window.size[1] / 2)}')
        self.tracker.sendCommand('online_dcorr_button = ON')
        self.tracker.sendCommand('normal_click_dcorr = OFF')

        # Drift correction for the eye tracker
        while not self.DUMMY:

            # terminate the task if no longer connected to the tracker or
            # user pressed Ctrl-C to terminate the task
            if (not self.tracker.isConnected()) or self.tracker.breakPressed():
                self.tracker.sendMessage('experiment_aborted')
                self.terminate_tracker()
                self.end()
                raise pylink.ABORT_EXPT

            # drift-check and re-do camera setup if ESCAPE is pressed
            try:
                error = self.tracker.doDriftCorrect(int(fix_x), int(fix_y), 1, 1)
                # break following a success drift-check
                if error is not pylink.ESC_KEY:
                    self._log_run("Tracker drift corrected.")
                    self.tracker.sendMessage("drift_corrected")
                    break
            except:
                self.tracker.sendMessage('drift_correction_failed')
                self._log_run("Tracker not drift corrected.")

    def terminate_tracker(self, end: bool = False):
        """
        Terminate the task gracefully and retrieve the EDF data file

        Parameters
        ----------
        end : whether shutdown psychopy and system program or not

        """
        # Check if there is an active connection
        if self.tracker.isConnected() or self.DUMMY:

            # Terminate the current trial first if the task terminated prematurely
            error = self.tracker.isRecording()
            if error:
                self.tracker.sendMessage("abort_trial")
                self.abort_trial()

            # Put tracker in Offline mode
            self.tracker.setOfflineMode()

            # Clear the Host PC screen and wait for 500 ms
            self.tracker.sendCommand('clear_screen 0')
            pylink.msecDelay(500)

            # Close the edf data file on the Host
            self.tracker.closeDataFile()

            # Show a file transfer message on the screen
            msg = 'EDF data is transferring from EyeLink Host PC...'
            self.show_msg(msg, wait_for_keypress=False)

            # Download the EDF data file from the Host PC to a local data folder
            # parameters: source_file_on_the_host, destination_file_on_local_drive
            try:
                self.tracker.receiveDataFile(self.files["edf_host"], self.files["edf_local"])
            except RuntimeError as error:
                logging.error('Error in downloading the EDF file:', error)

            # Close the link to the tracker.
            if end:
                pylink.closeGraphics()
                self.tracker.close()

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
        self.tracker.sendMessage(f'TRIAL_FAIL')

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
