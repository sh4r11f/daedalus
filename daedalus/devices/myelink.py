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
        self.tracked_eye = None
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

    def configure_init(self):
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

        if (
            (available == 0 and intended == "LEFT_EYE") or 
            (available == 1 and intended == "RIGHT_EYE") or 
            (available == 2 and intended == "BINOCULAR")
        ):
            if log:
                self.eyelink.sendMessage(f"EYE_USED {available} {intended}")
            self.tracked_eye = intended
            success = True
        else:
            self.error = f"Bad Eye. Available: {available}, Intended: {intended}"

        return success

    def get_eye_gaze_pos(self, sample):
        """
        Get the eye sample from the Eyelink tracker
        """
        if self.tracked_eye == "LEFT_EYE" and sample.isLeftSample():
            g_x, g_y = sample.getLeftEye().getGaze()
        elif self.tracked_eye == "RIGHT_EYE" and sample.isRightSample():
            g_x, g_y = sample.getRightEye().getGaze()
        else:
            g_x, g_y = None, None

        return g_x, g_y

    def online_event_monitor(self, event_list=None):
        """
        Find an event in the Eyelink tracker
        """
        if event_list is None:
            event_list = {
                "fixation_start": pylink.STARTFIX,
                "fixation_end": pylink.ENDFIX,
                "fixation_update": pylink.FIXUPDATE,
                "saccade_start": pylink.STARTSACC,
                "saccade_end": pylink.ENDSACC
            }

        info = {key: None for key in event_list.keys()}

        while True:
            event = self.eyelink.getNextData()

            if not event:
                break

            data = self.eyelink.getFloatData()
            if self.tracked_eye == data.getEye():
                if event == event_list["fixation_start"]:
                    info["fixatoin_start"] = self.detect_fixation_start_event(data)
                elif event == event_list["fixation_end"]:
                    info["fixation_end"] = self.detect_fixation_end_event(data)
                elif event == event_list["fixation_update"]:
                    info["fixation_update"] = self.detect_fixation_update_event(data)
                elif event == event_list["saccade_start"]:
                    info["saccade_start"] = self.detect_saccade_start_event(data)
                elif event == event_list["saccade_end"]:
                    info["saccade_end"] = self.detect_saccade_end_event(data)
                else:
                    pass

        return info

    def detect_fixation_start_event(self, data):
        """
        Process the fixation start event
        """
        time = data.getStartTime()
        gaze_x, gaze_y = data.getStartGaze()
        ppd_x, ppd_y = data.getStartPPD()

        return {
            "time": time,
            "gaze_x": gaze_x,
            "gaze_y": gaze_y,
            "ppd_x": ppd_x,
            "ppd_y": ppd_y
        }

    def detect_fixation_end_event(self, data):
        """
        Process the fixation end event
        """
        time = data.getEndTime()
        gaze_x, gaze_y = data.getEndGaze()
        ppd_x, ppd_y = data.getEndPPD()

        return {
            "time": time,
            "gaze_x": gaze_x,
            "gaze_y": gaze_y,
            "ppd_x": ppd_x,
            "ppd_y": ppd_y
        }

    def detect_fixation_update_event(self, data):
        """
        Process the fixation update event
        """
        time = data.getTime()
        gaze_x, gaze_y = data.getAverageGaze()
        ppd_start_x, ppd_start_y = data.getStartPPD()
        ppd_end_x, ppd_end_y = data.getEndPPD()
        ppd_x = (ppd_start_x + ppd_end_x) / 2
        ppd_y = (ppd_start_y + ppd_end_y) / 2

        return {
            "time": time,
            "gaze_x": gaze_x,
            "gaze_y": gaze_y,
            "ppd_x": ppd_x,
            "ppd_y": ppd_y
        }

    def detect_saccade_start_event(self, data):
        """
        Process the saccade start event
        """
        time = data.getStartTime()
        gaze_x, gaze_y = data.getStartGaze()
        ppd_x, ppd_y = data.getStartPPD()

        return {
            "time": time,
            "gaze_x": gaze_x,
            "gaze_y": gaze_y,
            "ppd_x": ppd_x,
            "ppd_y": ppd_y
        }

    def detect_saccade_end_event(self, data):
        """
        Process the saccade end event
        """
        time = data.getEndTime()
        gaze_x, gaze_y = data.getEndGaze()
        ppd_x, ppd_y = data.getEndPPD()

        return {
            "time": time,
            "gaze_x": gaze_x,
            "gaze_y": gaze_y,
            "ppd_x": ppd_x,
            "ppd_y": ppd_y
        }

    def online_event_processor(self, event_type: str):
        """
        Process the events in the Eyelink tracker

        Args:
            event_type (str): The type of event to process.
        """
        info = []
        prev_sample = None

        while True:
            buffer_item = self.eyelink.getNextData()
            if not buffer_item:
                break

            if buffer_item == pylink.FIXSTART:
                if prev_sample is None or buffer_item.getTime() != prev_sample.getTime():
                    prev_sample = buffer_item
                    if self.tracked_eye == "RIGHT_EYE" and buffer_item.isRightSample():
                        data = buffer_item.getRightEye()
                    elif self.tracked_eye == "LEFT_EYE" and buffer_item.isLeftSample():
                        data = buffer_item.getLeftEye()

                    if event_type == "fixation":
                        info.append(self.process_fixation_samples(data))
                    elif event_type == "saccade":
                        info.append(self.process_saccade_samples(data))
                    else:
                        raise NotImplementedError(f"Event type {event_type} is not implemented.")

        return info

    def process_fixation_end_event(self, event_data):
        """
        Process the fixation end event

        Args:
            event_data (pylink.Data): The fixation end event data.

        Returns:
            dict: The processed fixation end event data.
        """
        time_start = event_data.getStartTime()
        time_end = event_data.getEndTime()
        duration = time_end - time_start

        gaze_start_x, gaze_start_y = event_data.getStartGaze()
        gaze_end_x, gaze_end_y = event_data.getEndGaze()
        gaze_avg_x, gaze_avg_y = event_data.getAverageGaze()

        ppd_start_x, ppd_start_y = event_data.getStartPPD()
        ppd_end_x, ppd_end_y = event_data.getEndPPD()
        ppd_avg = (ppd_start_x + ppd_end_x) / 2, (ppd_start_y + ppd_end_y) / 2

        pupil_start = event_data.getStartPupilSize()
        pupil_end = event_data.getEndPupilSize()
        pupil_avg = event_data.getAveragePupilSize()

        return {
            "time_start": time_start,
            "time_end": time_end,
            "duration": duration,
            "gaze_start": (gaze_start_x, gaze_start_y),
            "gaze_end": (gaze_end_x, gaze_end_y),
            "gaze_avg": (gaze_avg_x, gaze_avg_y),
            "ppd_start": (ppd_start_x, ppd_start_y),
            "ppd_end": (ppd_end_x, ppd_end_y),
            "ppd_avg": ppd_avg,
            "pupil_start": pupil_start,
            "pupil_end": pupil_end,
            "pupil_avg": pupil_avg
        }

    def process_saccade_end_event(self, event_data):
        """
        Process the saccade end event

        Args:
            event_data (pylink.Data): The saccade end event data.

        Returns:
            dict: The processed saccade end event data.
        """
        time_start = event_data.getStartTime()
        time_end = event_data.getEndTime()
        duration = time_end - time_start

        gaze_start_x, gaze_start_y = event_data.getStartGaze()
        gaze_end_x, gaze_end_y = event_data.getEndGaze()

        ppd_start_x, ppd_start_y = event_data.getStartPPD()
        ppd_end_x, ppd_end_y = event_data.getEndPPD()

        velocity_start = event_data.getStartVelocity()
        velocity_end = event_data.getEndVelocity()
        velocity_avg = event_data.getAverageVelocity()
        velocity_peak = event_data.getPeakVelocity()

        amplitude = event_data.getAmplitude()
        angle = event_data.getAngle()

        return {
            "time_start": time_start,
            "time_end": time_end,
            "duration": duration,
            "gaze_start": (gaze_start_x, gaze_start_y),
            "gaze_end": (gaze_end_x, gaze_end_y),
            "ppd_start": (ppd_start_x, ppd_start_y),
            "ppd_end": (ppd_end_x, ppd_end_y),
            "velocity_start": velocity_start,
            "velocity_end": velocity_end,
            "velocity_avg": velocity_avg,
            "velocity_peak": velocity_peak,
            "amplitude": amplitude,
            "angle": angle
        }

    def online_sample_processor(self, event_type: str):
        """
        Process samples for each event in the link buffer (typically called at the end of each trial)

        Args:
            event_type (str): The type of event to process.
        """
        info = []
        prev_sample = None
        
        while True:
            buffer_item = self.eyelink.getNextData()
            if not buffer_item:
                break

            if buffer_item is not None and buffer_item.getType() == pylink.SAMPLE_TYPE:
                if prev_sample is None or buffer_item.getTime() != prev_sample.getTime():
                    prev_sample = buffer_item
                    if self.tracked_eye == "RIGHT_EYE" and buffer_item.isRightSample():
                        data = buffer_item.getRightEye()
                    elif self.tracked_eye == "LEFT_EYE" and buffer_item.isLeftSample():
                        data = buffer_item.getLeftEye()

                    if event_type == "fixation":
                        info.append(self.process_fixation_samples(data))
                    elif event_type == "saccade":
                        info.append(self.process_saccade_samples(data))
                    else:
                        raise NotImplementedError(f"Event type {event_type} is not implemented.")

        return info

    def process_fixation_samples(self, sample):
        """
        Process the fixation start event
        """
        # Get the sample info
        time = sample.getTime()
        gaze_x, gaze_y = sample.getGaze()
        ppd_x, ppd_y = sample.getPPD()
        pupil_size = sample.getPupilSize()

        return {
            "time": time,
            "gaze_x": gaze_x,
            "gaze_y": gaze_y,
            "ppd_x": ppd_x,
            "ppd_y": ppd_y,
            "pupil_size": pupil_size
        }

    def process_saccade_samples(self, sample):
        """
        Process the saccade start event
        """
        # Get the sample info
        time = sample.getTime()
        gaze_x, gaze_y = sample.getGaze()
        ppd_x, ppd_y = sample.getPPD()
        velocity = sample.getVelocity()

        return {
            "time": time,
            "gaze_x": gaze_x,
            "gaze_y": gaze_y,
            "ppd_x": ppd_x,
            "ppd_y": ppd_y,
            "velocity": velocity
        }

    def fixation_wait_event(self):
        """
        Method to detect fixation from the Eyelink over link.
        """
        info = None

        # Pump events until a fixation event is detected
        while True:

            # Get the event and break if no event is left/detected
            event = self.eyelink.getNextData()
            if not event:
                break

            # Check for fixation start event
            if event == pylink.STARTFIX:

                # Get the data
                data = self.eyelink.getFloatData()

                # Check eye and save data
                if self.tracked_eye == data.getEye():
                    time_fix = data.getTime()
                    time_offset = self.eyelink.trackerTimeOffset()
                    gaze_x, gaze_y = data.getStartGaze()
                    ppd_x, ppd_y = data.getStartPPD()

                    self.eyelink.sendMessage(f"FixStartTime_{time_fix}")
                    self.eyelink.sendMessage(f"FixStartGaze_{gaze_x}_{gaze_y}")
                    self.eyelink.sendMessage(f"FixStartPPD_{ppd_x}_{ppd_y}")

                    info = {
                        "time": time_fix,
                        "time_offset": time_offset,
                        "gaze": (gaze_x, gaze_y),
                        "ppd": (ppd_x, ppd_y)
                    }

                    break

        return info

    def fixation_monitor_event(self):
        """
        Method to monitor fixation from the Eyelink over link.
        """
        info = None

        # Pump events until a fixation update event is detected
        while True:

            # Get the event and break if no event is left/detected
            event = self.eyelink.getNextData()
            if not event:
                break

            # Check for fixation start event
            if event == pylink.FIXUPDATE:

                # Get the data
                data = self.eyelink.getFloatData()

                # Check eye and save data
                if self.tracked_eye == data.getEye():
                    time_fix = data.getTime()
                    time_offset = self.eyelink.trackerTimeOffset()
                    gaze_avg_x, gaze_avg_y = data.getAverageGaze()
                    ppd_start_x, ppd_start_y = data.getStartPPD()
                    ppd_end_x, ppd_end_y = data.getEndPPD()
                    ppd_avg_x = (ppd_start_x + ppd_end_x) / 2
                    ppd_avg_y = (ppd_start_y + ppd_end_y) / 2

                    self.eyelink.sendMessage(f"FixStartTime_{time_fix}")
                    self.eyelink.sendMessage(f"FixStartGaze_{gaze_avg_x}_{gaze_avg_y}")
                    self.eyelink.sendMessage(f"FixStartPPD_{ppd_avg_x}_{ppd_avg_y}")

                    info = {
                        "time": time_fix,
                        "time_offset": time_offset,
                        "gaze": (gaze_avg_x, gaze_avg_y),
                        "ppd": (ppd_avg_x, ppd_avg_y)
                    }

                    break

        return info

    def saccade_detect_event(self):
        """
        Method to detect saccade from the Eyelink over link.
        """
        info = {"saccade_start": None, "saccade_end": None}

        # Pump events until a saccade event is detected
        while True:

            # Get the event and break if no event is left/detected
            event = self.eyelink.getNextData()
            if not event:
                break

            # Check for saccade start event
            if event == pylink.STARTSACC:

                # Get the data
                data = self.eyelink.getFloatData()

                # Check eye and save data
                if self.tracked_eye == data.getEye():
                    time = data.getTime()
                    time_offset = self.eyelink.trackerTimeOffset()
                    gaze_x, gaze_y = data.getStartGaze()
                    ppd_x, ppd_y = data.getStartPPD()
                    velocity = data.getStartVelocity()

                    self.eyelink.sendMessage(f"SacStartTime_{time}")
                    self.eyelink.sendMessage(f"SacStartGaze_{gaze_x}_{gaze_y}")
                    self.eyelink.sendMessage(f"SacStartPPD_{ppd_x}_{ppd_y}")
                    self.eyelink.sendMessage(f"SacStartVel_{velocity}")

                    info["saccade_start"] = {
                        "time": time,
                        "saccade_start_gaze": (gaze_x, gaze_y),
                        "saccade_start_ppd": (ppd_x, ppd_y),
                        "saccade_start_velocity": velocity
                    }

            # Check for saccade end event
            elif event == pylink.ENDSACC:

                # Get the data
                data = self.eyelink.getFloatData()

                # Check eye and save data
                if self.tracked_eye == data.getEye():
                    start_gaze_x, start_gaze_y = data.getStartGaze()
                    end_gaze_x, end_gaze_y = data.getEndGaze()
                    ppd_x, ppd_y = data.getPPD()
                    start_time = data.getStartTime()
                    end_time = data.getEndTime()
                    time = data.getTime()
                    self.eyelink.sendMessage(f"FixStartTime_{time}")
                    info = {"time": time}
                    self.eyelink.sendMessage(f"FixStartGaze_{gaze_x}_{gaze_y}")
                    self.eyelink.sendMessage(f"FixStartPPD_{ppd_x}_{ppd_y}")
                    info["fixation_start_gaze"]["x"] = gaze_x
                    info["fixation_start_gaze"]["y"] = gaze_y
                    info["fixation_start_ppd"]["x"] = ppd_x
                    info["fixation_start_ppd"]["y"] = ppd_y
                    break

        return info

    def fixation_wait_realtime(self, target_pos, valid_dist, min_gaze_dur, clock):
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

    def fixation_monitor_realtime(self, target_pos, valid_dist, prev_sample):
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

    def saccade_detect_realtime(
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