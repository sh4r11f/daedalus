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

import pylink
from psychopy.tools.monitorunittools import deg2pix

from daedalus.utils import get_hypot


class MyeLink:
    """
    Class for controlling Eyelink 1000 eyetracker

    Args:
        exp_name (str): The name of the experiment.
        tracker_config (dict): The configuration of the tracker.
        dummy (bool): Whether to use the dummy mode.
        model (str): The model of the tracker.

    Attributes:
        exp_name (str): The name of the experiment.
        tracker_config (dict): The configuration of the tracker.
        dummy (bool): Whether to use the dummy mode.
        model (str): The model of the tracker.
        eyelink (pylink.EyeLink): The Eyelink tracker object.
        tracked_eye (str): The eye being tracked.
        error (str): The error message.
    """
    def __init__(self, exp_name, tracker_name, tracker_config, dummy):

        # Setup
        self.exp_name = exp_name
        self.tracker_name = tracker_name
        self.tracker_config = tracker_config
        self.dummy = dummy

        self.eyelink = None
        self.tracked_eye = None

    def delay(self, delay_time=None):
        """
        Delay the Eyelink tracker
        """
        if delay_time is None:
            delay_time = self.tracker_config["delay_time"]
        self.eyelink.pumpDelay(delay_time)

    def connect(self):
        """
        Connect to the Eyelink tracker

        Returns:
            str or None: The error message.
        """
        error = None
        if self.dummy:
            self.eyelink = pylink.EyeLink(None)
        else:
            try:
                self.eyelink = pylink.EyeLink(self.tracker_config["Connection"]["ip_address"])
                self.eyelink.sendMessage("Connection_OK")
                self.delay()
            except RuntimeError as err:
                error = err

        return error

    def terminate(self, host_file, display_file):
        """
        Close the connection to the Eyelink tracker

        Args:
            host_file (str): The file on the host.
            display_file (str): The file on the display.

        Returns:
            str or None: The error message.
        """
        error = None

        # Check connection
        if self.eyelink.isConnected():

            # Stop recording
            if self.eyelink.isRecording():
                # To get the last events in the buffer
                self.delay()
                self.eyelink.stopRecording()
                self.delay()
                self.eyelink.sendMessage("Recording_Stop")
                self.delay()

            # Send the tracker to offline mode
            self.go_offline()

            # Clear screen
            self.eyelink.sendCommand('clear_screen 0')
            self.delay()

            # Close the edf data file on the Host
            self.eyelink.closeDataFile()
            self.delay()

            # Download the EDF data file from the Host PC to the Display PC
            try:
                self.eyelink.receiveDataFile(host_file, display_file)
                self.eyelink.sendMessage("Download_OK")
            except RuntimeError as err:
                error = err
                self.eyelink.sendMessage("Download_Fail")
            self.delay()

            # Close the connection to the tracker
            self.eyelink.closeGraphics()
            err = self.eyelink.close()
            if err:
                error = err
        else:
            error = "DISCONNECTED"

        return error

    def open_edf_file(self, host_file):
        """
        Open a file to record the data and initialize it

        Args:
            host_file (str): The file on the host.

        Returns:
            str or None: The error message.
        """
        err = self.eyelink.openDataFile(host_file)  # Gives 0 or error code
        if not err:
            self.eyelink.sendCommand(f"add_file_preamble_text {self.tracker_config['preamble_text']}")
            self.delay()
            self.eyelink.sendMessage("EDF_OK")
            error = err
        else:
            self.eyelink.sendMessage("EDF_Fail")
            error = None
        self.delay()

        return error

    def configure(self):
        """
        Configure the Eyelink 1000 connection to track the specified events and have the appropriate setup
        e.g., sample rate and calibration type.
        """
        # Set the configuration from the config dictionary
        for key, value in self.tracker_config["AutoConfig"].items():
            self.eyelink.sendCommand(f"{key} = {value}")
            self.delay()

        self.eyelink.sendCommand("Configuration_OK")
        self.delay()

    def set_calibration_graphics(self, graph_env):
        """
        Set the calibration graphics

        Args:
            graph_env: External graphics environment to use for calibration
        """
        # Set window info
        width = int(self.tracker_config["screen_width"])
        height = int(self.tracker_config["screen_height"])

        self.eyelink.sendMesage(f"DISPLAY_COORDS 0 0 {width - 1} {height - 1}")
        self.delay()

        self.eyelink.sendMesage(f"screen_pixel_coords 0 0 {width - 1} {height - 1}")
        self.delay()

        if graph_env is None:
            bits = int(self.tracker_config["screen_bits"])
            self.eyelink.openGraphics((width, height), bits)
        else:
            self.eyelink.openGraphics(graph_env)

        self.eyelink.sendCommand("clear_screen 0")
        self.delay()

        self.eyelink.sendMessage("Graphics_OK")
        self.delay()

    def flush(self):
        """
        Flush the Eyelink buffer
        """
        self.eyelink.flushKeyButtons()
        self.eyelink.flushGetkeyQueue()
        self.resetData()
        self.delay()

        self.eyelink.sendMessage("Flush_OK")
        self.delay()

    def go_offline(self, wait=500):

        if self.eyelink.isRecording():
            self.delay()
            self.eyelink.stopRecording()
            self.delay()
            self.eyelink.sendMessage("Recording_Stop")
            self.delay()

        self.eyelink.setOfflineMode()
        err = self.eyelink.waitForModeChange(wait)
        if not err:
            self.eyelink.sendMessage("Offline_OK")
            error = err
        else:
            self.eyelink.sendMessage("Offline_Fail")
            error = "Failed to go offline."

        self.delay()

        return error

    def calibrate(self):
        """
        Calibrate the Eyelink 1000

        Returns:
            str or None: The error message.
        """
        try:
            # Start the calibration
            self.eyelink.doTrackerSetup()
            self.eyelink.sendMessage("Calibration_OK")
            error = None
        except RuntimeError as err:
            self.eyelink.exitCalibration()
            self.eyelink.sendMessage("Calibration_Fail")
            error = err

        self.delay()

        return error

    def routine_checks(self):
        """
        Perform routine checks on the Eyelink tracker

        Returns:
            str or None: The error message.
        """
        error = None
        if not self.eyelink.isConnected():
            self.eyelink.sendMessage("Connection_Fail")
            error = "DISCONNECTED"

        if self.eyelink.breakPressed():
            self.eyelink.sendMessage("Task_Terminated")
            error = "TERMINATED"

        return error

    def drift_correct(self, fix_pos=None):
        """
        Run drift correction

        Args:
            fix (tuple): The fixation point.

        Returns:
            str or None: The error message.
        """
        if fix_pos is None:
            monitor = pylink.getDisplayInformation()
            fx, fy = monitor.width / 2, monitor.height / 2
        else:
            fx, fy = fix_pos

        while True:
            # Check
            err = self.routine_checks()
            if err is not None:
                return err

            # Perform drift correction
            err = self.eyelink.doDriftCorrect(fx, fy, draw=1, allow_setup=0)
            # break if successful
            if err != pylink.ESC_KEY:
                break
            else:
                self.eyelink.sendMessage("Drift_Terminated")
                self.delay()
                err = "RECALIBRATE"

        return err

    def come_online(self):
        """
        Start recording the data

        Returns:
            str or None: The error message.
        """
        # Record samples and events and save to file and send over link
        err = self.eyelink.startRecording(1, 1, 1, 1)
        if not err:
            self.delay()
            self.eyelink.sendMessage("Recording_Start")
            error = None
        else:
            self.eyelink.sendMessage("Recording_Fail")
            error = err

        self.delay()

        return error

    def check_sample_rate(self):
        """
        Check the sample rate of the Eyelink tracker

        Returns:
            str or None: The error message.
        """
        tracker_rate = self.eyelink.getSampleRate()
        intended_rate = self.tracker_config["sample_rate"]

        if tracker_rate != intended_rate:
            error = f"Sample rate mismatch. Tracker: {tracker_rate}, Intended: {intended_rate}"
            self.eyelink.sendMessage("SampleRate_Fail")
        else:
            error = None
            self.eyelink.sendMessage("SampleRate_OK")

        self.delay()

        return error

    def check_eye(self, log=False):
        """
        Check if the eye is being tracked

        Args:
            log (bool): Whether to log the message.

        Returns:
            str or None: The error message.
        """
        # Get the eye information
        available = self.eyelink.eyeAvailable()
        intended = self.tracker_config["AutoConfig"]["active_eye"]

        if (
            (available == 0 and intended == "LEFT") or
            (available == 1 and intended == "RIGHT") or
            (available == 2 and intended == "BINOCULAR")
        ):
            self.tracked_eye = intended
            error = None
            if log:
                self.eyelink.sendMessage("Eye_OK")
        else:
            self.tracked_eye = None
            error = "BAD_EYE"
            self.eyelink.sendMessage("Eye_Fail")

        self.delay()

        return error

    def get_correct_eye_data(self, sample):
        """
        Get the correct eye data from a sample

        Args:
            sample (pylink.Sample): The sample data.

        Returns:
            pylink.EyeData or None: The eye data.
        """
        if self.tracked_eye == "LEFT" and sample.isLeftSample():
            data = sample.getLeftEye()
        elif self.tracked_eye == "RIGHT" and sample.isRightSample():
            data = sample.getRightEye()
        else:
            data = None

        return data

    def process_event_online(self, events_of_interest: dict = None):
        """
        Find an event in the Eyelink tracker

        Args:
            events_of_interest (dict): Events to monitor.

        Returns:
            dict: The event information.
        """
        if events_of_interest is None:
            events_of_interest = {
                "fixation_start": pylink.STARTFIX,
                "fixation_end": pylink.ENDFIX,
                "fixation_update": pylink.FIXUPDATE,
                "saccade_start": pylink.STARTSACC,
                "saccade_end": pylink.ENDSACC
            }

        # Initialize the event information
        info = {key: [] for key in events_of_interest.keys()}

        # Go through the samples and events in the buffer
        while True:

            err = self.routine_checks()
            if err is not None:
                return err
            
            item_type = self.eyelink.getNextData()
            if not item_type:
                break

            # Get the event data for the tracked eye
            if item_type in events_of_interest.values():
                event_name = [key for key, value in events_of_interest.items() if value == item_type][0]
                data = self.eyelink.getFloatData()
                if data.getEye() == self.tracked_eye:
                    if event_name == "fixation_start":
                        func = self.detect_fixation_start_event
                    elif event_name == "fixation_update":
                        func = self.detect_fixation_update_event
                    elif event_name == "saccade_start":
                        func = self.detect_saccade_start_event
                    elif event_name == "fixation_end":
                        func = self.process_fixation_end_event
                    elif event_name == "saccade_end":
                        func = self.process_saccade_end_event

                    info[event_name].append(func(data))

        return info

    def process_samples_online(self):
        """
        Process samples for each event in the link buffer (typically called at the end of each trial)

        Returns:
            list: A list of dictionaries containing the processed sample data.
        """
        info = []
        prev_sample = None

        while True:

            err = self.routine_checks()
            if err is not None:
                return err

            buffer_item = self.eyelink.getNextData()
            if not buffer_item:
                break

            if buffer_item is not None:
                if buffer_item.getType() == pylink.SAMPLE_TYPE:
                    if prev_sample is None or buffer_item.getTime() != prev_sample.getTime():

                        prev_sample = buffer_item
                        info = self.process_sample(buffer_item)

        return info

    def process_sample(self, sample):
        """
        Process the fixation start event
        """
        time = sample.getTime()
        ppd_x, ppd_y = sample.getPPD()

        eye_data = self.get_correct_eye_data(sample)
        if eye_data is not None:
            gaze_x, gaze_y = eye_data.getGaze()
            pupil_size = eye_data.getPupilSize()

        return {
            "time": time,
            "gaze_x": gaze_x,
            "gaze_y": gaze_y,
            "ppd_x": ppd_x,
            "ppd_y": ppd_y,
            "pupil_size": pupil_size
        }

    def detect_fixation_start_event(self, data):
        """
        Process the fixation start event
        """
        time = data.getStartTime()
        gaze_x, gaze_y = data.getStartGaze()
        ppd_x, ppd_y = data.getStartPPD()

        return {
            "time_start": time,
            "gaze_start_x": gaze_x,
            "gaze_start_y": gaze_y,
            "ppd_start_x": ppd_x,
            "ppd_start_y": ppd_y
        }

    def detect_fixation_end_event(self, data):
        """
        Process the fixation end event
        """
        time = data.getEndTime()
        gaze_x, gaze_y = data.getEndGaze()
        ppd_x, ppd_y = data.getEndPPD()

        return {
            "time_end": time,
            "gaze_end_x": gaze_x,
            "gaze_end_y": gaze_y,
            "ppd_end_x": ppd_x,
            "ppd_end_y": ppd_y
        }

    def detect_fixation_update_event(self, data):
        """
        Process the fixation update event
        """
        time_start = data.getStartTime()
        time_end = data.getEndTime()
        duration = time_end - time_start
        gaze_x, gaze_y = data.getAverageGaze()
        ppd_start_x, ppd_start_y = data.getStartPPD()
        ppd_end_x, ppd_end_y = data.getEndPPD()
        ppd_x = (ppd_start_x + ppd_end_x) / 2
        ppd_y = (ppd_start_y + ppd_end_y) / 2

        return {
            "time_start": time_start,
            "time_end": time_end,
            "duration": duration,
            "gaze_avg_x": gaze_x,
            "gaze_avg_y": gaze_y,
            "ppd_avg_x": ppd_x,
            "ppd_avg_y": ppd_y
        }

    def detect_saccade_start_event(self, data):
        """
        Process the saccade start event
        """
        time = data.getStartTime()
        gaze_x, gaze_y = data.getStartGaze()
        ppd_x, ppd_y = data.getStartPPD()

        return {
            "time_start": time,
            "gaze_start_x": gaze_x,
            "gaze_start_y": gaze_y,
            "ppd_start_x": ppd_x,
            "ppd_start_y": ppd_y
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
            "gaze_end_x": gaze_x,
            "gaze_end_y": gaze_y,
            "ppd_end_x": ppd_x,
            "ppd_end_y": ppd_y
        }

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
        ppd_avg_x, ppd_avg_y = (ppd_start_x + ppd_end_x) / 2, (ppd_start_y + ppd_end_y) / 2

        dva_avg_x, dva_avg_y = gaze_avg_x / ppd_avg_x, gaze_avg_y / ppd_avg_y

        pupil_start = event_data.getStartPupilSize()
        pupil_end = event_data.getEndPupilSize()
        pupil_avg = event_data.getAveragePupilSize()

        return {
            "time_start": time_start,
            "time_end": time_end,
            "duration": duration,
            "gaze_start_x": gaze_start_x,
            "gaze_start_y": gaze_start_y,
            "gaze_end_x": gaze_end_x,
            "gaze_end_y": gaze_end_y,
            "gaze_avg_x": gaze_avg_x,
            "gaze_avg_y": gaze_avg_y,
            "ppd_start_x": ppd_start_x,
            "ppd_start_y": ppd_start_y,
            "ppd_end_x": ppd_end_x,
            "ppd_end_y": ppd_end_y,
            "ppd_avg_x": ppd_avg_x,
            "ppd_avg_y": ppd_avg_y,
            "dva_avg_x": dva_avg_x,
            "dva_avg_y": dva_avg_y,
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
        ppd_avg_x, ppd_avg_y = (ppd_start_x + ppd_end_x) / 2, (ppd_start_y + ppd_end_y) / 2.0

        dva_avg_x, dva_avg_y = (gaze_end_x - gaze_start_x) / ppd_avg_x, (gaze_end_y - gaze_start_y) / ppd_avg_y
        amp_x, amp_y = event_data.getAmplitude()
        angle = event_data.getAngle()

        velocity_start = event_data.getStartVelocity()
        velocity_end = event_data.getEndVelocity()
        velocity_avg = event_data.getAverageVelocity()
        velocity_peak = event_data.getPeakVelocity()

        return {
            "time_start": time_start,
            "time_end": time_end,
            "duration": duration,
            "gaze_start_x": gaze_start_x,
            "gaze_start_y": gaze_start_y,
            "gaze_end_x": gaze_end_x,
            "gaze_end_y": gaze_end_y,
            "ppd_start_x": ppd_start_x,
            "ppd_start_y": ppd_start_y,
            "ppd_end_x": ppd_end_x,
            "ppd_end_y": ppd_end_y,
            "ppd_avg_x": ppd_avg_x,
            "ppd_avg_y": ppd_avg_y,
            "dva_avg_x": dva_avg_x,
            "dva_avg_y": dva_avg_y,
            "amp_x": amp_x,
            "amp_y": amp_y,
            "angle": angle,
            "velocity_start": velocity_start,
            "velocity_end": velocity_end,
            "velocity_avg": velocity_avg,
            "velocity_peak": velocity_peak
        }

    def realtime_fixation_monitor(self, target_pos, valid_dist, min_gaze_dur, clock):
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
            
            if not self.eyelink.isConnected():
                return "Disconnected from the tracker."

            if self.eyelink.breakPressed():
                return "User terminated the task."

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

    def realtime_fixation_detector(self, target_pos, valid_dist, prev_sample):
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
                gaze_x, gaze_y = self.get_eye_gaze_pos(self.tracked_eye, current_sample)
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

    def realtime_saccade_detector(
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

            if not self.eyelink.isConnected():
                return "Disconnected from the tracker."

            if self.eyelink.breakPressed():
                return "User terminated the task."

            # wait for a saccade event to occur
            if clock.getTime() - target_onset >= wait_duration:
                saccade_info["status"] = 0
                saccade_info["message"] = "saccade_time_out"
                self.tracker.sendMessage('SACCADE FAIL')
                self.tracker.sendMessage('saccade_time_out')
                break
            # grab the events in the buffer.
            eye_event = self.tracker.getNextData()
            if (eye_event is not None) and (eye_event == pylink.ENDSACC):
                eye_dat = self.tracker.getFloatData()
                if eye_dat.getEye() == self.params["EYE"]:
                    sac_amp = eye_dat.getAmplitude()  # amplitude
                    sac_start_time = eye_dat.getStartTime()  # onset time
                    sac_end_time = eye_dat.getEndTime()  # offset time
                    sac_start_pos = eye_dat.getStartGaze()  # start position
                    sac_end_pos = eye_dat.getEndGaze()  # end position

                    # ignore saccades occurred before target onset
                    if sac_start_time <= target_onset:
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
                        srt = sac_start_time - target_onset

                        # count as a correct response if the saccade lands in
                        # the square shaped hit region
                        land_err = [sac_end_pos[0] - target_x, sac_end_pos[1] - target_y]

                        # Check hit region and set accuracy
                        # if np.fabs(land_err[0]) < acc_region and np.fabs(land_err[1]) < acc_region:
                        if self.get_hypot(sac_end_pos, [target_x, target_y]) < valid_radius:
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
