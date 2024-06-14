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
import pandas as pd

from daedalus.devices.myelink import MyeLink
from daedalus import utils
import pylink
from psychopy import core
from .EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy
from .psyphy import PsychoPhysicsExperiment, PsychoPhysicsDatabase
from ..data.database import EyeTrackingEvent, EyeTrackingSample


class Eyetracking(PsychoPhysicsExperiment):
    """
    Eyetracking class for running experiments.

    Args:
        root (str or Path): The root directory of the project.
        platform (str): The platform where the experiment is running.
        debug (bool): Whether to run the experiment in debug mode or not.
    """
    def __init__(
        self,
        name: str,
        root,
        version: str,
        platform: str,
        debug: bool,
        tracker_model: str = "Eyelink1000Plus"
    ):
        # Setup
        super().__init__(name, root, version, platform, debug)

        self.exp_type = "eyetracking"

        # Tracker
        self.tracker_model = tracker_model
        self.tracker_params = utils.load_config(self.files["eyetrackers_params"])[self.tracker_model]
        self.tracker = None

        # Files
        self.events_data_file = None
        self.samples_data_file = None
        self.edf_display_file = None
        self.edf_host_file = None

        # Data
        self.events_data = None
        self.samples_data = None

    def init_experiment(self):
        """
        """
        subj_df = super().init_experiment()

        # Connect to the eye tracker
        self.init_tracker()

        # Set it to idle mode
        self.tracker.go_offline(1000)

        # Open the EDF file on the tracker
        err = self.tracker.open_edf_file()
        if err is not None:
            self.logger.critical(f"Failed to open EDF file: {err}")
            self.goodbye(err)

        # Open graphics environment
        ge = self.forge_graphics_env()
        self.tracker.set_calibration_graphics(ge)

        return subj_df

    def prepare_block(self, task_name, block_id, n_blocks, calib=False):
        """
        """
        self.init_block_data(task_name, block_id)
        self.block_clock = core.Clock()

        # Show block info as text
        txt = f"You are about to begin block {block_id}/{n_blocks}.\n\n"
        txt += "Press Space to start or Enter to recalibrate the eye tracker."
        resp = self.show_msg(txt)
        if resp in ["escape", "ctrl+c"]:
            self.logger.critical("User quit the experiment.")
            self.goodbye("User quit.")
        elif resp == "space":
            if calib:
                txt = "Recalibration is required!\n\n"
                txt += "Press Enter to recalibrate the eye tracker."
                resp = self.show_msg(txt)
                if resp == "escape":
                    self.logger.critical("User quit the experiment.")
                    self.goodbye("User quit.")
                elif resp == "enter":
                    self.logger.info("Recalibrating the eye tracker.")
                    self.run_calibration()
            self.logger.info(f"Starting block {block_id}.")
        elif resp == "enter":
            self.logger.info("Recalibrating the eye tracker.")
            self.run_calibration()

    def prepare_trial(self, trial_id, fix_pos=None):
        """
        """
        super().prepare_trial(trial_id)

        # Drift correction
        fix_pos = self.cart_to_mat(*fix_pos)
        err = self.tracker.drift_correct(fix_pos)
        if err is not None:
            if err == "RECALIBRATE":
                self.logger.info("Drift correction needs recalibration.")
            else:
                self.logger.critical(f"Drift correction failed: {err}")
            return err
        else:
            self.logger.info("Drift correction successful.")

        # Start recording
        err = self.tracker.come_online()
        if err is not None:
            self.logger.critical(f"Recording failed: {err}")
            return err
        else:
            self.logger.info("Recording started.")

    def trial_cleanup(self):
        """
        """
        # Stop recording
        self.tracker.go_offline()
        self.tracker.flush()

        # Clear the screen
        self.clear_screen()

    def block_cleanup(self):
        """
        """
        # Stop recording
        self.tracker.go_offline()
        self.tracker.flush()

        # Clear the screen
        self.clear_screen()

    def init_tracker(self):
        """
        Initializes the eye tracker and connect to it and configure it.

        Returns:
            MyeLink: The initialized eye tracker object.
        """
        self.tracker = MyeLink(self.name, self.tracker_params, self.debug)
        err = self.tracker.connect()
        if err is None:
            self.tracker.configure()
        else:
            self.logger.critical(f"Failed to connect to {self.tracker_model}: {err}")
            self.goodbye(err)

    def system_check(self):
        """
        """
        # Check the system
        warnings = super().system_check()

        # Start recording
        err = self.tracker.come_online()
        if err is not None:
            warnings.append("(✗) Eye tracker error in recording.")
            self.logger.warning(f"Recording error: {err}")
            return warnings

        # Check the eye
        err = self.tracker.check_eye()
        if err is not None:
            warnings.append("(✗) Eye mismatch.")
            self.logger.critical(f"Eye mismatch: {err}")
            return warnings

        # Cache some samples
        t = 300
        self.tracker.delay(t)

        # Check the samples
        samples = self.tracker.process_samples_online()
        if samples:
            warnings.append(f"(✓) Eye tracker is working. Got {len(samples)} samples in {t}ms.")
            self.logger.info(f"Eye tracker is working. Got {len(samples)} samples in {t}ms.")
        else:
            warnings.append(f"(✗) Eye tracker is not working. Got 0 samples in {t}ms.")
            self.logger.warning(f"Eye tracker is not working. Got 0 samples in {t}ms.")

        # Set the tracker to idle mode
        self.tracker.go_offline()

        return warnings

    def forge_graphics_env(self):
        """
        Sets up the graphics environment for the calibration.

        Returns:
            EyeLinkCoreGraphicsPsychoPy: The graphics environment for the calibration.
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

        return genv

    def check_fixation_in_square(self, center, radius, fix_coords):
        """
        Check if the fixation is within the square.

        Args:
            center (tuple): The center of the square.
            radius (int): The radius of the square.
            fix_coords (tuple): The fixation coordinates.

        Returns:
            bool: Whether the fixation is within the square or not.
        """
        valid_x = center[0] - radius < fix_coords[0] < center[0] + radius
        valid_y = center[1] - radius < fix_coords[1] < center[1] + radius

        if valid_x and valid_y:
            return True
        else:
            return False

    def check_fixation_in_circle(self, fx, fy, gx, gy, radius):
        """
        Check if the fixation is within the circle.

        Args:
            fx (float): The x-coordinate of the fixation.
            fy (float): The y-coordinate of the fixation.
            gx (float): The x-coordinate of the gaze.
            gy (float): The y-coordinate of the gaze.
            radius (float): The radius of the circle.

        Returns:
            bool: Whether the fixation is within the circle or not.
        """
        offset = utils.get_hypot(fx, fy, gx, gy)
        if offset < radius:
            return True
        else:
            return False

    def mat_to_cart(self, x, y):
        """
        Center the Eyelink coordinates.

        Args:
            x (float): The x-coordinate.
            y (float): The y-coordinate.

        Returns:
            tuple: The centered coordinates.
        """
        return x - self.window.size[0] / 2, self.window.size[1] / 2 - y

    def cart_to_mat(self, x, y):
        """
        Convert center coordinates to matrix coordinates.

        Args:
            x (float): The x-coordinate.
            y (float): The y-coordinate.

        Returns:
            tuple: The matrix coordinates.
        """
        return self.window.size[0] / 2 + x, self.window.size[1] / 2 - y

    def establish_fixation(self, fixation, timeout, radi, method="circle"):
        """
        Establish fixation.

        Args:
            fixation (object): The fixation stimulus.
            timeout (float): The time to wait for fixation.
            radi (float): The radius of the circle.

        Returns:
            bool: Whether the fixation is established or not.
        """
        events_of_interest = {"fixation_start": pylink.STARTFIX}
        start_time = self.trial_clock.getTime()
        fx, fy = fixation.pos
        fixated = False

        while not fixated:
            if self.trial_clock.getTime() - start_time > timeout:
                self.logger.critical("Fixation timeout.")
                self.tracker.eyelink.sendMessage("Fixation_Timeout")
                break

            fixation.draw()
            self.window.flip()

            eye_events = self.tracker.detect_event_online(events_of_interest)
            if eye_events["fixation_start"]:
                gaze_x = eye_events["fixation_start"][-1]["gaze_start_x"]
                gaze_y = eye_events["fixation_start"][-1]["gaze_start_y"]
                gx, gy = self.mat_to_cart(gaze_x, gaze_y)
                if method == "circle":
                    valid = self.check_fixation_in_circle(fx, fy, gx, gy, radi)
                else:
                    raise NotImplementedError("Only circle method is implemented.")
                if valid:
                    fixated = True
                    self.logger.info("Fixation established.")
                    self.tracker.eyelink.sendMessage("Fixation_OK")

        return fixated

    def run_calibration(self):
        """
        """
        # Message
        msg = "In the next screen, press `c` to calibrate the eyetracker.\n\n"
        msg += "Once the calibration is done, press `Enter` to accept the calibration and resume the experiment.\n\n"
        msg += "Press Space to continue..."
        self.show_msg(msg)

        self.tracker.go_offline()
        error = self.tracker.calibrate()
        if error is not None:
            self.logger.critical(f"Calibration failed: {error}")
        else:
            self.logger.info("Calibration successful.")

        return error

    def monitor_fixation(self, fix_pos, radi, method="circle"):
        """
        Monitor fixation.

        Args:
            fixation (object): The fixation stimulus.
            timeout (float): The time to wait for fixation.
            radi (float): The radius of the circle.

        Returns:
            bool: Whether the fixation is established or not.
        """
        fx, fy = fix_pos
        events_of_interest = {"fixation_update": pylink.FIXUPDATE, "fixation_end": pylink.ENDFIX}
        events = self.tracker.detect_event_online(events_of_interest)

        if events["fixation_end"]:
            self.logger.critical("Fixation lost.")
            self.tracker.eyelink.sendMessage("Fixation_Fail")
            return False

        valid_updates = []
        if events["fixation_update"]:
            for event in events["fixation_update"]:
                gaze_x = event["gaze_x"]
                gaze_y = event["gaze_y"]
                gx, gy = self.mat_to_cart(gaze_x, gaze_y)
                if method == "circle":
                    check = self.check_fixation_in_circle(fx, fy, gx, gy, radi)
                else:
                    raise NotImplementedError("Only circle method is implemented.")
                valid_updates.append(check)

        if all(valid_updates):
            fixated = True
            self.logger.info("Fixation updated.")
            self.tracker.eyelink.sendMessage("Fixation_OK")
        else:
            fixated = False
            self.logger.critical("Fixation lost.")
            self.tracker.eyelink.sendMessage("Fixation_Fail")

        return fixated

    def init_session_dirs(self, subj_id, ses_id):
        """
        """
        # Create directories for the session
        super().init_session_dirs(subj_id, ses_id)

        # Add eyetracking specific directories
        eye_data_dir = self.ses_data_dir / "eye"
        eye_data_dir.mkdir(parents=True, exist_ok=True)

    def _init_events_file(self, file_path):
        """
        """
        columns = [
            "TrialIndex",
        ]

        df = pd.DataFrame(columns=columns)
        df.to_csv(file_path, index=False)

    def _init_samples_file(self, file_path):
        """
        """
        columns = [
            "TrialIndex",
        ]

        df = pd.DataFrame(columns=columns)
        df.to_csv(file_path, index=False)

    def init_block_data(self, task_name, block_id):
        """
        """
        super().init_block_data(task_name, block_id)

        # Add eyetracking specific files
        events_file = self.ses_data_dir / "eye" / f"sub-{self.subj_id}_ses-{self.ses_id}_task-{task_name}_block-{block_id}_EyeEvents.csv"
        if events_file.exists():
            self.logger.critical(f"File {events_file} already exists. Making a backup and removing the file.")
            backup_file = events_file.with_suffix(".bak")
            events_file.rename(backup_file)
        self._init_events_file(events_file)
        self.events_data_file = events_file
        self.events_data = pd.read_csv(events_file)

        samples_file = self.ses_data_dir / "eye" / f"sub-{self.subj_id}_ses-{self.ses_id}_task-{task_name}_block-{block_id}_EyeSamples.csv"
        if samples_file.exists():
            self.logger.critical(f"File {samples_file} already exists. Making a backup and removing the file.")
            backup_file = samples_file.with_suffix(".bak")
            samples_file.rename(backup_file)
        self._init_samples_file(samples_file)
        self.samples_data_file = samples_file
        self.samples_data = pd.read_csv(samples_file)

        edf_display_file = self.ses_data_dir / "eye" / f"sub-{self.subj_id}_ses-{self.ses_id}_task-{task_name}_block-{block_id}.edf"
        if edf_display_file.exists():
            self.logger.critical(f"File {edf_display_file} already exists. Making a backup and removing the file.")
            backup_file = edf_display_file.with_suffix(".bak")
            edf_display_file.rename(backup_file)
        self.edf_display_file = edf_display_file

        self.edf_host_file = f"sub-{self.subj_id}_ses-{self.ses_id}_task-{task_name}_block-{block_id}_host.edf"


class EyeTrackingDatabase(PsychoPhysicsDatabase):
    """
    Derived class for handling eye-tracking specific database operations.
    """
    def __init__(self, db_path):
        """
        Initialize the database connection and create tables for eye-tracking data.
        """
        super().__init__(db_path)
        self.exp_type = "eyetracking"

    def add_eye_tracking_event(self, trial_id, event_type, time_start, time_end, **kwargs):
        """
        Add a new eye-tracking event to the database.

        Args:
            trial_id (int): The ID of the trial.
            event_type (str): The type of the event.
            time_start (float): The start time of the event.
            time_end (float): The end time of the event.
            duration (float): The duration of the event.
            **kwargs: Additional optional parameters.

        Returns:
            int: The ID of the added eye-tracking event.
        """
        session = self.Session()
        event = EyeTrackingEvent(
            trial_id=trial_id,
            event_type=event_type,
            time_start=time_start,
            time_end=time_end,
            **kwargs
        )
        session.add(event)
        session.commit()
        return event.uid

    def add_eye_tracking_sample(self, event_id, timestamp, gaze_x, gaze_y, ppd_x, ppd_y, pupil):
        """
        Add a new eye-tracking sample to the database.

        Args:
            event_id (int): The ID of the eye-tracking event.
            timestamp (float): The timestamp of the sample.
            gaze_x (float): The x-coordinate of the gaze.
            gaze_y (float): The y-coordinate of the gaze.
            ppd_x (float): The x-coordinate of the pupil position.
            ppd_y (float): The y-coordinate of the pupil position.
            pupil (float): The size of the pupil.
            **kwargs: Additional optional parameters.

        Returns:
            int: The ID of the added eye-tracking sample.
        """
        session = self.Session()
        sample = EyeTrackingSample(
            event_id=event_id, timestamp=timestamp, gaze_x=gaze_x, gaze_y=gaze_y, ppd_x=ppd_x, ppd_y=ppd_y, pupil=pupil
        )
        session.add(sample)
        session.commit()
        return sample.uid

    def get_eye_tracking_events(self, trial_id):
        """
        Retrieve all eye-tracking events for a given trial.

        Args:
            trial_id (int): The ID of the trial.

        Returns:
            list: List of eye-tracking events.
        """
        session = self.Session()
        return session.query(EyeTrackingEvent).filter_by(trial_id=trial_id).all()

    def get_eye_tracking_samples(self, event_id):
        """
        Retrieve all eye-tracking samples for a given event.

        Args:
            event_id (int): The ID of the eye-tracking event.

        Returns:
            list: List of eye-tracking samples.
        """
        session = self.Session()
        return session.query(EyeTrackingSample).filter_by(event_id=event_id).all()
    
    # def run_calibration(self):
    #     """
    #     Calibrate the Eyelink 1000
    #     """
    #     # A guiding message
    #     msg = 'Press ENTER and C to recalibrate the tracker.\n\n' \
    #           'Once the calibration is done, press ENTER and O to resume the experiment.'
    #     self.show_msg(msg)

    #     # Initiate calibration
    #     try:
    #         self.tracker.doTrackerSetup()
    #         self._log_run("Tracker calibrated.")
    #         self.tracker.sendMessage('tracker_calibrated')

    #     except (RuntimeError, AttributeError) as err:
    #         self._log_run(f"Tracker not calibrated: {err}")
    #         self.tracker.exitCalibration()

    # def abort_trial(self):
    #     """Ends recording and recycles the trial"""

    #     # Stop recording
    #     if self.tracker.isRecording():
    #         # add 100 ms to catch final trial events
    #         pylink.pumpDelay(100)
    #         self.tracker.stopRecording()

    #     # clear the screen
    #     self.clear_screen()

    #     # Send a message to clear the Data Viewer screen
    #     bgcolor_RGB = (116, 116, 116)
    #     self.tracker.sendMessage('!V CLEAR %d %d %d' % bgcolor_RGB)

    #     # send a message to mark trial end
    #     # self.tracker.sendMessage(f'TRIAL_RESULT {pylink.TRIAL_ERROR}')
    #     self.tracker.sendMessage('TRIAL_FAIL')

    #     # Log
    #     self._log_run("!!! Trial aborted.")

    # def validate_all_gaze(self, gaze_arr: Union[List, np.ndarray], period: str):
    #     """
    #     Checks all gaze reports and throws an error if there is any gaze deviation.
    #     The runtime error should be caught later and handled by recycling the trial.

    #     Parameters
    #     ----------
    #     gaze_arr : list or array
    #         The array of True or False for each gaze time point.

    #     period : str
    #         Name of the period where the validation is being done for.
    #     """
    #     # Check all the timepoints
    #     if not all(gaze_arr):

    #         # Log
    #         self._log_run(f"Gaze fail in trial {self.trial}, {period} period.")
    #         self.tracker.sendMessage(f"{period.upper()}_FAIL")

    #         # Change fixation color to red
    #         self.stimuli["fix"].color = [1, -1, -1]
    #         self.stimuli["fix"].draw()
    #         self.window.flip()

    #         # Throw an error
    #         raise RuntimeError

    # def setup_run(self, inst_msg: str):
    #     """
    #     Initiates clocks, shows beginning message, and logs start of a run.

    #     Parameters
    #     ----------
    #     inst_msg : str
    #         A text that has instructions for start of the experiment
    #     """
    #     # Setup files for logging and saving this run
    #     self._file_setup()
    #     self.files["tracker_local"] = str(self.files["run"]) + '.edf'
    #     # initiate file on the tracker
    #     self.open_tracker_file()

    #     # Clock
    #     self.clocks["run"].reset()

    #     # Change the log level task name for this run
    #     logging.addLevel(99, self.task)

    #     # Log the start of the run
    #     self.log_section('Run', 'start')
    #     self.tracker.sendMessage(f"TASK_START")
    #     self.tracker.sendMessage(f"TASKID_{self.task}")
    #     self.tracker.sendMessage(f"RUN_START")
    #     self.tracker.sendMessage(f"RUNID_{self.exp_run}")

    #     # Show instruction message before starting
    #     self.show_msg(inst_msg)

    # def setup_block(self, calib=True):
    #     """
    #     Sets up an experimental block. Shows a text message and initiates and calibrates the eye tracker.
    #     """
    #     # Timing
    #     self.clocks["block"].reset()

    #     # Tracker initialization
    #     self.tracker.setOfflineMode()
    #     if calib:
    #         self.run_calibration()

    #     # clear the host screen
    #     self.tracker.sendCommand('clear_screen 0')

    #     # log the beginning of block
    #     self.log_section('Block', 'start')
    #     self.tracker.sendMessage(f"BLOCK_START")
    #     self.tracker.sendMessage(f"BLOCKID_{self.block}")

    # def trial_cleanup(self):
    #     """
    #     Turns off the stimuli and stop the tracker to clean up the trial.
    #     """
    #     # Turn off all stimuli
    #     self.clear_screen()
    #     # clear the host screen too
    #     self.tracker.sendCommand('clear_screen 0')

    #     # Stop recording frame interval
    #     self.window.recordFrameIntervals = False

    #     # Stop recording; add 100 msec to catch final events before stopping
    #     pylink.pumpDelay(100)
    #     self.tracker.stopRecording()

    # def block_cleanup(self):
    #     """ Logs the end of a block and shows message about the remaining blocks"""

    #     # Log the end of the block
    #     self.log_section("Block", "end")
    #     self.tracker.sendMessage(f"BLOCKID_{self.block}")
    #     self.tracker.sendMessage(f"BLOCK_END")

    #     # Turn everything off
    #     self.clear_screen()
    #     # clear the host screen
    #     self.tracker.sendCommand('clear_screen 0')

    #     # Get the number of all blocks for this task
    #     n_blocks = int(self.TASKS[self.task]["blocks"])

    #     # Show a message between the blocks
    #     remain_blocks = n_blocks - self.block
    #     btw_blocks_msg = f"You finished block number {self.block}.\n\n" \
    #                      f"{remain_blocks} more block(s) remaining in this run.\n\n" \
    #                      "When you are ready, press the SPACEBAR to continue to calibration."
    #     self.show_msg(btw_blocks_msg)

    # def run_cleanup(self, remain_runs):
    #     """ Housekeeping at the end of a run"""

    #     # Log
    #     self.log_section("Run", "end")
    #     self.tracker.sendMessage(f"RUNID_{self.exp_run}")
    #     self.tracker.sendMessage(f"RUN_END")
    #     self.tracker.sendMessage(f"TASKID_{self.task}")
    #     self.tracker.sendMessage(f"TASK_END")

    #     # Turn everything off
    #     self.clear_screen()
    #     self.tracker.sendCommand('clear_screen 0')

    #     # Show an info message
    #     btw_msg = f"\U00002705 You finished run number {self.run} of the experiment \U00002705\n\n"
    #     if not self.practice:
    #         btw_msg += f"There are {remain_runs} run remaining!\n\n"
    #         btw_msg += "When you are ready, press the SPACEBAR to continue to calibration!"

    #     end_msg = "\U0000235F\U0000235F\U0000235F \n\n"
    #     end_msg += "Experiment ended! Thank you for your participation :-)\n\n"
    #     end_msg += "\U0000235F\U0000235F\U0000235F"

    #     # Terminate the tracker on the last run
    #     if remain_runs:
    #         self.show_msg(btw_msg, wait_for_keypress=False)
    #         self.terminate_tracker(end=False)
    #     else:
    #         self.show_msg(end_msg, wait_for_keypress=False)
    #         self.terminate_tracker(end=True)

    # def save_run_data(self, blocks: list, col_names: list):
    #     """
    #     Saves all the blocks and logs to disk.

    #     Parameters
    #     ----------
    #     blocks : list
    #         All numpy arrays that contain the experiment data

    #     col_names : list
    #         Column names for each trial block

    #     """
    #     # Compile all blocks into one giant array
    #     exp_data = np.concatenate(blocks)

    #     # Make a data frame
    #     pd_file = str(self.files["run"]) + '.csv'
    #     df = pd.DataFrame(exp_data, columns=col_names)

    #     # Save it
    #     df.to_csv(pd_file, sep=',', index=False)

    #     # Save the log files
    #     # logs are in a tuple (or list) with the first item being the log level and the second one the log file
    #     # self.logs["task"].write(self.files["run_log"])
    #     # get rid of this log file
    #     logging.flush()
    #     logging.root.removeTarget(self.logs["task"])
