#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================================== #
#
#
#                    SCRIPT: eylink.py
#
#
#          DESCRIPTION: Class for controlling Eyelink tracker
#
#
#                       RULE: DAYW
#
#
#
#                  CREATOR: Sharif Saleki
#                         TIME: 05-27-2024-[78 105 98 105114117]
#                       SPACE: Dartmouth College, Hanover, NH
#
# ==================================================================================================== #
from typing import Union, List, Tuple, Dict

import numpy as np

from psychopy.tools.monitorunittools import deg2pix
import pylink
from daedalus.utils import get_hypot


class MyeLink:
    """
    Class for controlling Eyelink 1000 eyetracker
    """
    def __init__(self, tracker_config, exp_name, exp_window, dummy):

        # Setup
        self.tracker_config = tracker_config
        self.exp_name = exp_name
        self.exp_window = exp_window
        self.dummy = dummy

        self.eyelink = None
        self.error = None

    def connect(self):
        """
        Connect to the Eyelink tracker
        """
        if self.dummy:
            self.eyelink = pylink.EyeLink(None)
        else:
            try:
                self.eyelink = pylink.EyeLink(self.tracker_config["Connection"]["ip_address"])
            except RuntimeError as err:
                self.error = err
                return False
        return True

    def terminate(self, host_file, display_file):
        """
        Close the connection to the Eyelink tracker
        """
        success = False
        # Check connection
        if self.eyelink.isConnected():
            # Check recording status
            if self.eyelink.isRecording():
                pylink.pumpDelay(100)
                self.eyelink.stopRecording()

            # Set the tracker to offline mode
            self.eyelink.setOfflineMode()
            pylink.msecDelay(500)
            self.eyelink.sendCommand('clear_screen 0')
            pylink.msecDelay(500)
            
            # Close the edf data file on the Host
            self.eyelink.closeDataFile()
            try:
                # Download the EDF data file from the Host PC to the Display PC
                self.eyelink.receiveDataFile(host_file, display_file)
                # Close the connection to the Eyelink tracker
                self.eyelink.close()
                self.eyelink.closeGraphics()
                success = True
            except RuntimeError as err:
                self.error = err
        else:
            self.error = "Eyelink tracker is not connected"
        
        return success

    def open_file(self, host_file):
        """
        Open a file to record the data and initialize it
        """
        success = False
        try:
            # Open a files to record the data
            self.eyelink.openDataFile(host_file)
            # Add a header text to the EDF file to identify the current experiment name
            self.eyelink.sendCommand(f"add_file_preamble_text {self.exp_name}")
            success = True
        except RuntimeError as err:
            self.error = err
            
        return success

    def get_version(self):
        """
        Get the version of the Eyelink tracker
        """
        return self.eyelink.getTrackerVersion()

    def configure(self):
        """
        Configure the Eyelink 1000 connection to track the specified events and have the appropriate setup
        e.g., sample rate and calibration type.
        """
        # Set the tracker parameters
        self.eyelink.setOfflineMode()

        # Set the configuration from the config dict (which is a dictionary of dictionaries    )
        for config_type, config_dict in self.tracker_config.items():
            if config_type != "Connection":
                for key, value in config_dict.items():
                    self.eyelink.sendCommand(f"{key} = {value}")

        # Rest of the setup
        win_width_idx = self.exp_window.size[0] - 1
        win_height_idx = self.exp_window.size[1] - 1
        self.eyelink.sendCommnad(f"screen_pixel_coords = 0 0 {win_width_idx} {win_height_idx}")
        self.eyelink.sendMessage(f"DISPLAY_COORDS = {self.tracker_config['display_coords']}")

    def calibrate(self):
        """
        Calibrate the Eyelink 1000
        """
        success = False
        eye_correct = self.check_eye(log=True)
        if eye_correct:
            try:
                # Check and log the eye information
                # Start the calibration
                self.eyelink.doTrackerSetup()
                self.eyelink.sendMessage("tracker_calibrated")
                success = True
            except RuntimeError as err:
                self.eyelink.exitCalibration()
                self.error = err
        
        return success

    def check_eye(self, log=False):
        """
        Check if the eye is being tracked

        Args:
            log (bool): Whether to log the message.

        Returns:
            bool: Whether the eye is being tracked or not.
        """
        success = False
        available = self.eyelink.eyeAvailable()
        intended = self.tracker_config["General"]["active_eye"]

        if ((available == 0 and intended == "LEFT") or
            (available == 1 and intended == "RIGHT") or
            (available == 2 and intended == "BINOCULAR")):
            if log:
                self.eyelink.sendMessage(f"EYE_USED {available} {intended}")
            success = True
        else:
            self.error = f"Bad Eye. Available: {available}, Intended: {intended}"

        return success

    def get_eye_gaze_pos(self, tracked_eye, sample):
        """
        Get the eye sample from the Eyelink tracker
        """
        if tracked_eye == 0 and sample.isLeftSample():
            g_x, g_y = sample.getLeftEye().getGaze()
        elif tracked_eye == 1 and sample.isRightSample():
            g_x, g_y = sample.getRightEye().getGaze()
        else:
            g_x, g_y = None, None

        return g_x, g_y

    def check_fixation(self, target_pos, valid_dist, prev_sample):
        """
        Function to monitor gaze on a static point, like a fixation cross.

        The point of this function is to allow for checking fixation on every `n` frame(s).
        (n is controlled by the experiment script).

        Args:
            target_pos (tuple): The position of the fixation point.
            valid_dist (float): The radius of the gaze region in degrees.
            prev_sample (pylink.Sample): The previous sample.

        Returns:
            tuple:
                pylink.Sample: The new sample.
                dict: The fixation information containing gaze position, fixation status, and offset.
        """
        # Get the eye information
        if self.check_eye:
            tracked_eye = self.eyelink.eyeAvailable()

        # Define hit region
        valid_radius = deg2pix(valid_dist, self.monitor)
        target_x, target_y = target_pos
        fixation_info = None

        # Look at the buffer
        current_sample = self.eyelink.getNewestSample()
        if current_sample is not None:
            if (prev_sample is None) or (current_sample.getTime() != prev_sample.getTime()):
                # check if the new sample has data for the eye
                # currently being tracked; if so, we retrieve the current
                # gaze position and PPD (how many pixels correspond to 1
                # deg of visual angle, at the current gaze position)
                gaze_x, gaze_y = self.get_eye_gaze_pos(tracked_eye, current_sample)
                if (gaze_x is not None) and (gaze_y is not None):
                    # See if the current gaze position is in a region around the screen centered
                    offset = get_hypot(target_x, target_y, gaze_x, gaze_y)
                    fixating = offset < valid_radius

                    # Save the fixation information
                    fixation_info = {
                        "gaze_x": gaze_x,
                        "gaze_y": gaze_y,
                        "fixating": fixating,
                        "offset": offset
                    }

        return current_sample, fixation_info

    def wait_for_fixation(self, target_pos, valid_dist, min_gaze_dur, clock):
        """
        Runs a while loop and keeps it running until gaze on some region is established.

        Args:
            target_pos (tuple): The position of the target (e.g., fixation point).
            valid_dist (float): The radius of the gaze region in degrees.
            min_gaze_dur (int): The duration of gaze in milliseconds.
            clock (psychopy.core.Clock): The clock to keep track of time.

        Returns:
            bool: Whether the gaze is on the target for the minimum duration.
        """
        # Get the eye information
        if self.check_eye:
            tracked_eye = self.eyelink.eyeAvailable()

        # Define the hit region
        gaze_start_time = -1
        in_region = False
        region_radius = deg2pix(valid_dist, self.monitor)
        target_x, target_y = target_pos

        # Running the gaze loop
        trigger = self.dummy  # Trigger status: False until gaze is on target, True if in dummy mode
        prev_sample = None

        while not trigger:
            current_sample = self.tracker.getNewestSample()

            # check if the new sample has data for the eye currently being tracked
            if current_sample is not None:
                if (prev_sample is None) or (current_sample.getTime() != prev_sample.getTime()):
                    # set as the previous sample
                    prev_sample = current_sample

                    # get the gaze position
                    gaze_x, gaze_y = self.get_eye_gaze_pos(tracked_eye, current_sample)

                    # break the while loop if the current gaze position in the target region
                    if (gaze_x is not None) and (gaze_y is not None):
                        offset = get_hypot(target_x, target_y, gaze_x, gaze_y)
                        if offset < region_radius:
                            # start recording the gaze duration if not already started
                            if (not in_region) and (gaze_start_time == -1):
                                in_region = True
                                gaze_start_time = clock.getTime()
                            # if already recording check the gaze duration and fire
                            else:
                                gaze_dur = clock.getTime() - gaze_start_time
                                if gaze_dur > min_gaze_dur:
                                    trigger = True
                                    self.tracker.sendMessage("fixation_acquired")
                        # gaze outside the hit region, reset variables
                        else:
                            in_region = False
                            gaze_start_time = -1

        return trigger

    def check_saccade(
        self,
        target_onset: float,
        target_pos: Union[List, Tuple],
        valid_dist: float,
        wait_duration: int,
        clock
    ) -> Dict:
        """
        Checks if a saccade is made to a target position.

        Args:
            target_onset (float): The time at which the target appeared.
            target_pos (Union[List, Tuple]): The position of the target.
            valid_dist (float): The radius of the gaze region in degrees.
            wait_duration (int): The maximum time to wait for a saccade in ms. 
            clock (psychopy.core.Clock): The clock to keep track of time.

        Returns:
            dict:
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
        target_x = target_pos[0] + self.window.size[0] / 2.0
        target_y = target_pos[1] + self.window.size[1] / 2.0

        # Saccade checking variables
        saccade = self.dummy
        saccade_info = {"status": int(saccade)}
        # sac_start_time = -1
        # srt = -1  # initialize a variable to store saccadic reaction time (SRT)
        # land_err = -1  # landing error of the saccade
        # acc = 0  # hit the correct region or not
        valid_radius = deg2pix(valid_dist, self.monitor)
        amp_thresh = 3

        # Wait for the saccade
        while not saccade:
            # wait for a saccade event to occur
            if clock.getTime() - target_onset >= wait_duration:
                saccade_info["status"] = 0
                saccade_info["message"] = "saccade_time_out"
                self.tracker.sendMessage('SACCADE FAIL')
                self.tracker.sendMessage('saccade_time_out')
                break
            # grab the events in the buffer.
            eye_event = self.tracker.getNextData()
            if (eye_ev is not None) and (eye_ev == pylink.ENDSACC):
                eye_dat = self.tracker.getFloatData()
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
                        self.tracker.sendMessage("SACCADE_START")

                        # log a message to mark the time at which a saccadic
                        # response occurred; note that, here we are detecting a
                        # saccade END event; the saccade actually occurred some
                        # msecs ago. The following message has an additional
                        # time offset, so Data Viewer knows when exactly the
                        # "saccade_resp" event actually happened
                        t_offset = int(self.tracker.trackerTime() - sac_start_time)
                        sac_response_msg = f'{t_offset} saccade_resp'
                        self.tracker.sendMessage(sac_response_msg)
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