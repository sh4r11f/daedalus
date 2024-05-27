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

from .psyphy import PsychophysicsExperiment


class EyetrackingExperiment(PsychophysicsExperiment):
    """
    Main template for eye tracking experiments running on the Eyelink1000 Plus. It mostly adds functionality for
    communicating and controlling the Eyelink. Other than that, it pretty much implements all its parents methods.

    Most of the code in the methods are from examples in Pylink docs.

    Parameters
    ----------
    debug : bool
        Debug mode for the experiment.

    dummy_tracker : bool
        Debug mode for the eye tracker.

    root_dir : str or Path
        Home directory of the experiment.
    """
    def __init__(self, project_root: Union[str, Path], platform: str, debug: bool):

        # Setup
        super().__init__(project_root, platform, debug)

        # Eye tracking experiments always use pixel measurements.
        self.window.units = 'pix'
        # NOTE: If we don't do this early, the eyetracker will override it and we always get a grey background.
        self.window.color = self.exp_params["Environment"]["background_color"]

        # Add Eyelink-specific .edf files' paths and names
        self.files["edf_host"] = f"{self.NAME}{self._SUBJECT['id']:02d}.edf"

        # Initialize the eye tracker
        self.el_tracker = pylink.getEYELINK()

        # Connect and configure the tracker
        self.connect_tracker()
        self.open_tracker_file()
        self.configure_tracker()
        self.setup_calibration()

    def calibrate_tracker(self):
        """
		Calibrate the Eyelink 1000
		"""
  		# Configure a graphics environment (genv) for tracker calibration
        genv = EyeLinkCoreGraphicsPsychoPy(self.tracker, self.window)

		# Set background and foreground colors for the calibration target
		# in PsychoPy, (-1, -1, -1)=black, (1, 1, 1)=white, (0, 0, 0)=mid-gray
        foreground_color = (1, 1, 1)
        background_color = self.params["BG_COLOR"]
		# background_color = (-1, -1, -1)
        genv.setCalibrationColors(foreground_color, background_color)

		# Set up the calibration target
		# Use the default calibration target ('circle')
        genv.setTargetType('circle')

		# Configure the size of the calibration target (in pixels)
		# this option applies only to "circle" and "spiral" targets
        genv.setTargetSize(24)

		# Beeps to play during calibration, validation and drift correction
		# parameters: target, good, error
		#     target -- sound to play when target moves
		#     good -- sound to play on successful operation
		#     error -- sound to play on failure or interruption
		# Each parameter could be ''--default sound, 'off'--no sound, or a wav file
        genv.setCalibrationSounds('', '', '')

		# Request Pylink to use the PsychoPy window we opened above for calibration
        pylink.openGraphicsEx(genv)
		# Start the calibration
        self.tracker.doTrackerSetup()
  
    def connect_tracker(self):
        """
        Connects to an Eyelink 1000 using its python API (pylink).
        """
        # Initiate connection (only if not in dummy mode)
        if self.debug:
            self.el_tracker = pylink.EyeLink(None)
        else:
            # IP: 100.1.1.1
            try:
                self.el_tracker = pylink.EyeLink("100.1.1.1")
                self.logger.info("Tracker connected.")
            except RuntimeError as err:
                self.logger.critical(f'Error in connecting to the Eyelink: {err}')
                self.end()

    def open_tracker_file(self):
        """
        Creates an edf file on the host Eyelink computer to record the data
        """
        # Open an EDF data file on the Host PC
        try:
            self.el_tracker.openDataFile(self.files["edf_host"])
            logging.exp("Tracker file created.")
            # Add a header text to the EDF file to identify the current experiment name
            self.el_tracker.sendCommand(f"add_file_preamble_text {self.NAME}")

        except RuntimeError as err:
            logging.error('Error in creating the EDF file:', err)

            # close the link if we have one open
            if self.el_tracker.isConnected():
                self.el_tracker.close()

            # quit the experiment
            self.end()

    def configure_tracker(self):
        """
        Configure the Eyelink 1000 connection to track the specified events and have the appropriate setup
        e.g., sample rate and calibration type.
        """
        # Put the tracker in offline mode before we change tracking parameters
        self.el_tracker.setOfflineMode()

        # Get the software version:  1-EyeLink I, 2-EyeLink II, 3/4-EyeLink 1000, 5-EyeLink 1000 Plus, 6-Portable DUO
        eyelink_ver = 0  # set version to 0, in case running in Dummy mode
        if not self.DUMMY:
            vstr = self.el_tracker.getTrackerVersionString()
            eyelink_ver = int(vstr.split()[-1].split('.')[0])
            # print out some version info in the shell
            msg = f'Running experiment on {vstr}, version {eyelink_ver}'
            logging.info(msg)

        # File and Link data control
        # what eye events to save in the EDF file, include everything by default
        file_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT'

        # what eye events to make available over the link, include everything by default
        link_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,FIXUPDATE,INPUT'

        # what sample data to save in the EDF data file and to make available
        # over the link, include the 'HTARGET' flag to save head target sticker
        # data for supported eye trackers
        if eyelink_ver > 3:
            file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,HTARGET,GAZERES,BUTTON,STATUS,INPUT'
            link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,HTARGET,STATUS,INPUT'
        else:
            file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,GAZERES,BUTTON,STATUS,INPUT'
            link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT'

        # Send the parameters to the tracker
        self.el_tracker.sendCommand(f"file_event_filter = {file_event_flags}")
        self.el_tracker.sendCommand(f"file_sample_data = {file_sample_flags}")
        self.el_tracker.sendCommand(f"link_event_filter = {link_event_flags}")
        self.el_tracker.sendCommand(f"link_sample_data = {link_sample_flags}")

        # Optional tracking parameters
        # Sample rate, 250, 500, 1000, or 2000, check your tracker specification
        self.el_tracker.sendCommand("sample_rate 1000")

        # Choose a calibration type, H3, HV3, HV5, HV13 (HV = horizontal/vertical),
        if self.DEBUG:
            self.el_tracker.sendCommand("calibration_type = HV3")
        else:
            self.el_tracker.sendCommand("calibration_type = HV5")

        # Log
        logging.exp("Tracker configured.")

    def setup_calibration(self):
        """
        Prepares the Eyelink for calibration procedure.
        This includes setting the correct coordinates for the tracker and creating a graphics environment for
        calibration.
        """
        # get the native screen resolution used by PsychoPy
        scn_width, scn_height = self.window.size

        # Pass the display pixel coordinates (left, top, right, bottom) to the tracker
        # see the EyeLink Installation Guide, "Customizing Screen Settings"
        el_coords = f"screen_pixel_coords = 0 0 {scn_width - 1} {scn_height - 1}"
        self.el_tracker.sendCommand(el_coords)

        # Write a DISPLAY_COORDS message to the EDF file
        # Data Viewer needs this piece of info for proper visualization, see Data
        # Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
        dv_coords = f"DISPLAY_COORDS  0 0 {scn_width - 1} {scn_height - 1}"
        self.el_tracker.sendMessage(dv_coords)

        # Configure a graphics environment (genv) for tracker calibration
        genv = EyeLinkCoreGraphicsPsychoPy(self.el_tracker, self.window)
        msg = f"Version number of the CoreGraphics library: {genv}"
        logging.info(msg)

        # Set background and foreground colors for the calibration target
        # in PsychoPy, (-1, -1, -1)=black, (1, 1, 1)=white, (0, 0, 0)=mid-gray
        foreground_color = (1, 1, 1)
        background_color = self.params["BG_COLOR"]
        # background_color = (-1, -1, -1)
        genv.setCalibrationColors(foreground_color, background_color)

        # Set up the calibration target
        # Use the default calibration target ('circle')
        genv.setTargetType('circle')

        # Configure the size of the calibration target (in pixels)
        # this option applies only to "circle" and "spiral" targets
        genv.setTargetSize(24)

        # Beeps to play during calibration, validation and drift correction
        # parameters: target, good, error
        #     target -- sound to play when target moves
        #     good -- sound to play on successful operation
        #     error -- sound to play on failure or interruption
        # Each parameter could be ''--default sound, 'off'--no sound, or a wav file
        genv.setCalibrationSounds('', '', '')

        # Request Pylink to use the PsychoPy window we opened above for calibration
        pylink.openGraphicsEx(genv)

    def get_eye(self) -> int:
        """
        Checks if the tracker can find an eye, and whether that eye is the same as the one specified in the
        parameters file of the experiment.

        Returns
        -------
        int
            The eye used for the experiment: 0 is left and 1 is right.
        """
        # Get the eye info from tracker
        eye_used = self.el_tracker.eyeAvailable()

        # Get the eye info from parameters file
        eye_intend = int(self.params["EYE"])

        # Check the right eye
        if eye_used == 1 and eye_used == eye_intend:
            logging.info("Tracker using the right eye.")
            self.el_tracker.sendMessage("EYE_USED 1 RIGHT")

        # Check the left eye or detecting both eyes
        elif eye_used == 0 or eye_used == 2:
            eye_used = 0  # use the left eye only
            logging.info("Tracker using the left eye.")
            self.el_tracker.sendMessage("EYE_USED 0 LEFT")

        # If there is invalid eye
        else:
            msg = 'Error in getting eye information. Recalibrate the tracker.'
            logging.warning(msg)
            self.show_msg(msg)

        return eye_used

    def check_gaze(self, old_sample, fix_pos: Union[List, Tuple]):
        """
        Function to monitor gaze on a static point, like a fixation cross.

        Parameters
        ----------
        old_sample : instance of Sample type
            Previous eye position sample

        fix_pos : list or tuple
            The position of the fixation spot on the screen that needs to be checked for gaze info

        Returns
        -------
        tuple
            eye sample: instance of Sample type
            fixating: bool value of fixation check
        """
        # Get the eye information
        eye_used = self.el_tracker.eyeAvailable()

        # Define hit region
        region = deg2pix(float(self.params["GAZE_REGION"]), self.monitor)
        fix_x, fix_y = fix_pos
        g_x = None
        g_y = None

        # Have some faith!
        fixating = True

        # Do we have a sample in the sample buffer?
        # and does it differ from the one we've seen before?
        new_sample = self.el_tracker.getNewestSample()
        if new_sample is not None:
            if old_sample is not None:
                if new_sample.getTime() != old_sample.getTime():

                    # check if the new sample has data for the eye
                    # currently being tracked; if so, we retrieve the current
                    # gaze position and PPD (how many pixels correspond to 1
                    # deg of visual angle, at the current gaze position)
                    if eye_used == 1 and new_sample.isRightSample():
                        g_x, g_y = new_sample.getRightEye().getGaze()
                    if eye_used == 0 and new_sample.isLeftSample():
                        g_x, g_y = new_sample.getLeftEye().getGaze()

                    # See if the current gaze position is in a region around the screen centered
                    # if np.fabs(g_x - fix_x) < region and np.fabs(g_y - fix_y) < region:
                    if self.get_hypot([g_x, g_y], [fix_x, fix_y]) < region:
                        fixating = True
                    else:  # gaze outside the hit region
                        fixating = False

        return new_sample, fixating

    def wait_for_gaze(self, fix_pos: Union[List, Tuple]):
        """
        Runs a while loop and keeps it running until gaze on some region is established.

        Parameters
        ----------
        fix_pos : tuple or list
            Coordinates of the fixation spot to be monitored
        """
        # Clear cached PsychoPy events
        event.clearEvents()

        # Samples
        old_sample = None

        # Trigger status
        if self.DUMMY:
            trigger_fired = True
        else:
            trigger_fired = False

        # Fire the trigger following a 500-ms gaze
        min_dur = .5

        # The position of the gaze
        gaze_start = -1

        # Get the eye information
        eye_used = self.el_tracker.eyeAvailable()

        # Define the hit region
        in_hit_region = False
        region = deg2pix(float(self.params["GAZE_REGION"]), self.monitor)
        fix_x, fix_y = fix_pos

        # Running the gaze loop
        while not trigger_fired:

            # Do we have a sample in the sample buffer?
            # and does it differ from the one we've seen before?
            new_sample = self.el_tracker.getNewestSample()

            if new_sample is not None:
                if old_sample is not None:
                    if new_sample.getTime() != old_sample.getTime():

                        # check if the new sample has data for the eye
                        # currently being tracked; if so, we retrieve the current
                        # gaze position and PPD (how many pixels correspond to 1
                        # deg of visual angle, at the current gaze position)
                        if eye_used == 1 and new_sample.isRightSample():
                            g_x, g_y = new_sample.getRightEye().getGaze()
                        if eye_used == 0 and new_sample.isLeftSample():
                            g_x, g_y = new_sample.getLeftEye().getGaze()

                        # break the while loop if the current gaze position in a region
                        # if np.fabs(g_x - fix_x) < region and np.fabs(g_y - fix_y) < region:
                        if self.get_hypot([g_x, g_y], [fix_x, fix_y]) < region:
                            # record gaze start time
                            if not in_hit_region:
                                if gaze_start == -1:
                                    gaze_start = core.getTime()
                                    in_hit_region = True

                            # check the gaze duration and fire
                            if in_hit_region:
                                gaze_dur = core.getTime() - gaze_start
                                if gaze_dur > min_dur:
                                    trigger_fired = True
                                    self._log_run("Fixation detected.")
                                    self.el_tracker.sendMessage('gaze_confirmed')

                        # gaze outside the hit region, reset variables
                        else:
                            in_hit_region = False
                            gaze_start = -1

                # update the "old_sample"
                old_sample = new_sample

    def check_saccade(self, t_onset: float, target_pos: Union[List, Tuple], t_thresh: int = 500) -> Dict:
        """
        Checks if a saccade is made to a target position.

        Parameters
        ----------
        t_onset : float
            time of the onset of the target

        target_pos : list or tuple
            Coordinates of the target (0 centered around the center of the screen)

        t_thresh : int
            Duration to wait for saccade in ms

        Returns
        -------
        dict
            Information about the saccade:
                - status: made successfully or not
                - t_start: time of initiation
                - t_end: time of landing
                - amp_x: amplitude in X direction
                - amp_y: amplitude in Y direction
                - start_x: X of start position
                - start_y: Y of start position
                - end_x: X of landing
                - end_y: Y of landing
                - rt: saccadic reaction time
                - error_x: distance from the target in X
                - error_y: distance from the target in Y
                - accuracy: whether saccade was close to the target or not

        """
        # Set up the coordinates
        target_x, target_y = target_pos
        target_x = target_x + self.window.size[0] / 2.0
        target_y = target_y + self.window.size[1] / 2.0

        # Saccade checking vars
        if self.DUMMY:
            got_sac = True
            _info = {"status": 1}
        else:
            got_sac = False
            _info = None

        # sac_start_time = -1
        # srt = -1  # initialize a variable to store saccadic reaction time (SRT)
        # land_err = -1  # landing error of the saccade
        # acc = 0  # hit the correct region or not
        acc_region = deg2pix(1, self.monitor)
        amp_thresh = 3

        # Clear all cached events if there are any
        event.clearEvents()

        # Wait for the saccade
        while not got_sac:

            # wait for a maximum of 1000 milliseconds for the saccade to occur
            if self.el_tracker.trackerTime() - t_onset >= t_thresh:
                self.el_tracker.sendMessage('saccade_time_out')
                self.el_tracker.sendMessage('SACCADE_FAIL')
                self._log_run("Saccade timeout.")
                break

            # grab the events in the buffer, for more details,
            # see the example script "link_event.py"
            eye_ev = self.el_tracker.getNextData()
            if (eye_ev is not None) and (eye_ev == pylink.ENDSACC):
                eye_dat = self.el_tracker.getFloatData()
                if eye_dat.getEye() == self.params["EYE"]:
                    sac_amp = eye_dat.getAmplitude()  # amplitude
                    sac_start_time = eye_dat.getStartTime()  # onset time
                    sac_end_time = eye_dat.getEndTime()  # offset time
                    sac_start_pos = eye_dat.getStartGaze()  # start position
                    sac_end_pos = eye_dat.getEndGaze()  # end position

                    # ignore saccades occurred before target onset
                    if sac_start_time <= t_onset:
                        sac_start_time = -1
                        pass

                    # A correct saccade was initiated
                    elif np.hypot(sac_amp[0], sac_amp[1]) > amp_thresh:

                        # Saccade status
                        got_sac = True
                        self.el_tracker.sendMessage("SACCADE_START")

                        # log a message to mark the time at which a saccadic
                        # response occurred; note that, here we are detecting a
                        # saccade END event; the saccade actually occurred some
                        # msecs ago. The following message has an additional
                        # time offset, so Data Viewer knows when exactly the
                        # "saccade_resp" event actually happened
                        t_offset = int(self.el_tracker.trackerTime() - sac_start_time)
                        sac_response_msg = f'{t_offset} saccade_resp'
                        self.el_tracker.sendMessage(sac_response_msg)
                        srt = sac_start_time - t_onset

                        # count as a correct response if the saccade lands in
                        # the square shaped hit region
                        land_err = [sac_end_pos[0] - target_x, sac_end_pos[1] - target_y]

                        # Check hit region and set accuracy
                        # if np.fabs(land_err[0]) < acc_region and np.fabs(land_err[1]) < acc_region:
                        if self.get_hypot(sac_end_pos, [target_x, target_y]) < acc_region:
                            acc = 1
                        else:
                            acc = 0

                        # Save everything
                        _info = {
                            "status": got_sac,
                            "t_start": sac_start_time,
                            "t_end": sac_end_time,
                            "amp_x": sac_amp[0],
                            "amp_y": sac_amp[1],
                            "start_x": sac_start_pos[0],
                            "start_y": sac_start_pos[1],
                            "end_x": sac_end_pos[0],
                            "end_y": sac_end_pos[1],
                            "rt": srt,
                            "error_x": land_err[0],
                            "error_y": land_err[1],
                            "accuracy": acc
                        }
        return _info

    def run_calibration(self):
        """Calibrate the Eyelink 1000"""

        # A guiding message
        msg = 'Press ENTER and C to recalibrate the tracker.\n\n' \
              'Once the calibration is done, press ENTER and O to resume the experiment.'
        self.show_msg(msg)

        # Initiate calibration
        try:
            self.el_tracker.doTrackerSetup()
            self._log_run("Tracker calibrated.")
            self.el_tracker.sendMessage('tracker_calibrated')

        except (RuntimeError, AttributeError) as err:
            self._log_run(f"Tracker not calibrated: {err}")
            self.el_tracker.exitCalibration()

    def start_recording(self):
        """Initiate recording of eye data by the tracker"""

        try:
            # arguments: sample_to_file, events_to_file, sample_over_link,
            # event_over_link (1-yes, 0-no)
            self.el_tracker.startRecording(1, 1, 1, 1)
            self._log_run("Tracker recording.")
            self.el_tracker.sendMessage("recording")

        except RuntimeError as err:
            self._log_run(f"Tracker not recording: {err}")
            self.el_tracker.sendMessage("not_recording")
            return pylink.TRIAL_ERROR

    def check_connection(self):
        """Checks if the tracker is still alive"""

        # Get the status
        error = self.el_tracker.isConnected()  # returns 1 if connected, 0 if not connected, -1 if simulated

        # Check it only in the actual running mode because simulated tracker returns -1 in isConnected()
        if not self.DUMMY:

            # For some reason TRIAL_OK is 0 in pylink
            if error is pylink.TRIAL_OK:
                # Log the disconnect
                self.el_tracker.sendMessage('tracker_disconnected')
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
        self.el_tracker.sendCommand('driftcorrect_cr_disable = OFF')
        self.el_tracker.sendCommand(f'online_dcorr_refposn {int(self.window.size[0] / 2)},{int(self.window.size[1] / 2)}')
        self.el_tracker.sendCommand('online_dcorr_button = ON')
        self.el_tracker.sendCommand('normal_click_dcorr = OFF')

        # Drift correction for the eye tracker
        while not self.DUMMY:

            # terminate the task if no longer connected to the tracker or
            # user pressed Ctrl-C to terminate the task
            if (not self.el_tracker.isConnected()) or self.el_tracker.breakPressed():
                self.el_tracker.sendMessage('experiment_aborted')
                self.terminate_tracker()
                self.end()
                raise pylink.ABORT_EXPT

            # drift-check and re-do camera setup if ESCAPE is pressed
            try:
                error = self.el_tracker.doDriftCorrect(int(fix_x), int(fix_y), 1, 1)
                # break following a success drift-check
                if error is not pylink.ESC_KEY:
                    self._log_run("Tracker drift corrected.")
                    self.el_tracker.sendMessage("drift_corrected")
                    break
            except:
                self.el_tracker.sendMessage('drift_correction_failed')
                self._log_run("Tracker not drift corrected.")

    def terminate_tracker(self, end: bool = False):
        """
        Terminate the task gracefully and retrieve the EDF data file

        Parameters
        ----------
        end : whether shutdown psychopy and system program or not

        """
        # Check if there is an active connection
        if self.el_tracker.isConnected() or self.DUMMY:

            # Terminate the current trial first if the task terminated prematurely
            error = self.el_tracker.isRecording()
            if error:
                self.el_tracker.sendMessage("abort_trial")
                self.abort_trial()

            # Put tracker in Offline mode
            self.el_tracker.setOfflineMode()

            # Clear the Host PC screen and wait for 500 ms
            self.el_tracker.sendCommand('clear_screen 0')
            pylink.msecDelay(500)

            # Close the edf data file on the Host
            self.el_tracker.closeDataFile()

            # Show a file transfer message on the screen
            msg = 'EDF data is transferring from EyeLink Host PC...'
            self.show_msg(msg, wait_for_keypress=False)

            # Download the EDF data file from the Host PC to a local data folder
            # parameters: source_file_on_the_host, destination_file_on_local_drive
            try:
                self.el_tracker.receiveDataFile(self.files["edf_host"], self.files["edf_local"])
            except RuntimeError as error:
                logging.error('Error in downloading the EDF file:', error)

            # Close the link to the tracker.
            if end:
                pylink.closeGraphics()
                self.el_tracker.close()

    def abort_trial(self):
        """Ends recording and recycles the trial"""

        # Stop recording
        if self.el_tracker.isRecording():
            # add 100 ms to catch final trial events
            pylink.pumpDelay(100)
            self.el_tracker.stopRecording()

        # clear the screen
        self.clear_screen()

        # Send a message to clear the Data Viewer screen
        bgcolor_RGB = (116, 116, 116)
        self.el_tracker.sendMessage('!V CLEAR %d %d %d' % bgcolor_RGB)

        # send a message to mark trial end
        # self.el_tracker.sendMessage(f'TRIAL_RESULT {pylink.TRIAL_ERROR}')
        self.el_tracker.sendMessage(f'TRIAL_FAIL')

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
            self.el_tracker.sendMessage(f"{period.upper()}_FAIL")

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
        self.files["edf_local"] = str(self.files["run"]) + '.edf'
        # initiate file on the tracker
        self.open_tracker_file()

        # Clock
        self.clocks["run"].reset()

        # Change the log level task name for this run
        logging.addLevel(99, self.task)

        # Log the start of the run
        self.log_section('Run', 'start')
        self.el_tracker.sendMessage(f"TASK_START")
        self.el_tracker.sendMessage(f"TASKID_{self.task}")
        self.el_tracker.sendMessage(f"RUN_START")
        self.el_tracker.sendMessage(f"RUNID_{self.exp_run}")

        # Show instruction message before starting
        self.show_msg(inst_msg)

    def setup_block(self, calib=True):
        """
        Sets up an experimental block. Shows a text message and initiates and calibrates the eye tracker.
        """
        # Timing
        self.clocks["block"].reset()

        # Tracker initialization
        self.el_tracker.setOfflineMode()
        if calib:
            self.run_calibration()

        # clear the host screen
        self.el_tracker.sendCommand('clear_screen 0')

        # log the beginning of block
        self.log_section('Block', 'start')
        self.el_tracker.sendMessage(f"BLOCK_START")
        self.el_tracker.sendMessage(f"BLOCKID_{self.block}")

    def trial_cleanup(self):
        """
        Turns off the stimuli and stop the tracker to clean up the trial.
        """
        # Turn off all stimuli
        self.clear_screen()
        # clear the host screen too
        self.el_tracker.sendCommand('clear_screen 0')

        # Stop recording frame interval
        self.window.recordFrameIntervals = False

        # Stop recording; add 100 msec to catch final events before stopping
        pylink.pumpDelay(100)
        self.el_tracker.stopRecording()

    def block_cleanup(self):
        """ Logs the end of a block and shows message about the remaining blocks"""

        # Log the end of the block
        self.log_section("Block", "end")
        self.el_tracker.sendMessage(f"BLOCKID_{self.block}")
        self.el_tracker.sendMessage(f"BLOCK_END")

        # Turn everything off
        self.clear_screen()
        # clear the host screen
        self.el_tracker.sendCommand('clear_screen 0')

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
        self.el_tracker.sendMessage(f"RUNID_{self.exp_run}")
        self.el_tracker.sendMessage(f"RUN_END")
        self.el_tracker.sendMessage(f"TASKID_{self.task}")
        self.el_tracker.sendMessage(f"TASK_END")

        # Turn everything off
        self.clear_screen()
        self.el_tracker.sendCommand('clear_screen 0')

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

    @staticmethod
    def get_hypot(target, fix):

        fx, fy = np.fabs(fix)
        tx, ty = np.fabs(target)

        return np.hypot((tx - fx), (ty - fy))
