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
from daedalus.codes import Codex


class MyeLink:
    """
    Class for controlling Eyelink 1000 eyetracker

    Args:
        exp_name (str): The name of the experiment.
        tracker_name (str): The name of the tracker.
        tracker_config (dict): The configuration of the tracker.
        dummy (bool): Whether to use the dummy mode.

    Attributes:
        exp_name (str): The name of the experiment.
        tracker_name (str): The name of the tracker.
        params (dict): The configuration of the tracker.
        dummy (bool): Whether to use the dummy mode.
        eyelink (pylink.EyeLink): The Eyelink tracker.
        tracked_eye (str): The eye being tracked.
        codex (daedalus.codes.Codex): The codex for the tracker.
    """
    def __init__(self, exp_name, tracker_name, tracker_config, dummy):

        # Setup
        self.exp_name = exp_name
        self.tracker_name = tracker_name
        self.params = tracker_config
        self.dummy = dummy
        self.codex = Codex()

        if self.dummy:
            self.eyelink = pylink.EyeLink(None)
        else:
            self.eyelink = pylink.EyeLink(self.params["Connection"]["ip_address"])

    def delay(self, delay_time=None):
        """
        Delay the Eyelink tracker with pumpDelay

        Args:
            delay_time (float): The delay time.
        """
        if delay_time is None:
            delay_time = self.params["delay_time"]
        pylink.pumpDelay(delay_time)

    def short_delay(self, time=None):
        """
        Short delay for the Eyelink tracker

        Args:
            time (float): The delay time.
        """
        if time is None:
            time = self.params["short_delay"]
        self.eyelink.msecDelay(time)

    def codex_msg(self, proc, state, delay=True):
        """
        Send a code to the Eyelink tracker

        Args:
            proc (str): The process.
            state (str): The state.
            delay (bool): Whether to delay.

        Returns:
            str: The code.
        """
        msg = self.codex.message(proc, state)
        self.eyelink.sendMessage(msg)
        if delay:
            self.delay()

        return msg

    def direct_msg(self, msg, delay=True):
        """
        Send a message to the Eyelink tracker

        Args:
            msg (str): The message to send.
            delay (bool): Whether to delay.
        """
        self.eyelink.sendMessage(msg)
        if delay:
            self.delay()

    def configure(self, display_name):
        """
        Configure the Eyelink 1000 connection to track the specified events and have the appropriate setup
        e.g., sample rate and calibration type.

        Args:
            display_name (str): The name of the display PC.
        """
        # Enter setup menu
        # self.eyelink.doTrackerSetup()
        # Set the configuration from the config dictionary
        for key, value in self.params["AutoConfig"].items():
            self.eyelink.sendCommand(f"{key} = {value}")
            self.delay()
        self.eyelink.setName(display_name)
        self.codex_msg("config", "done")

    def go_offline(self):
        """
        Set the Eyelink tracker to offline mode
        """
        mode = self.eyelink.getCurrentMode()
        if mode != pylink.IN_IDLE_MODE:
            # Stop recording if it is still recording
            rec = self.eyelink.isRecording()
            if rec == pylink.TRIAL_OK:
                self.delay()
                self.eyelink.stopRecording()
                self.delay()
                self.codex_msg("rec", "stop")
            # Set the tracker to offline mode
            self.eyelink.setOfflineMode()
            self.codex_msg("idle", "init")

    def set_calibration_graphics(self, width, height, graphics_env):
        """
        Set the calibration graphics

        Args:
            graph_env: External graphics environment to use for calibration
        """
        # Set the display coordinates
        self.send_cmd(f"screen_pixel_coords = 0 0 {width - 1} {height - 1}")
        # log the display coordinates
        self.direct_msg(f"DISPLAY_COORDS 0 0 {width - 1} {height - 1}")
        # Set the calibration graphics
        pylink.openGraphicsEx(graphics_env)
        self.delay()

    def send_cmd(self, cmd):
        """
        Send a command to the Eyelink tracker

        Args:
            cmd (str): The command to send.
        """
        self.eyelink.sendCommand(cmd)
        self.delay()

    def come_online(self):
        """
        Start recording the data

        Returns:
            str or None: The error message.
        """
        # Record samples and events and save to file and send over link
        self.eyelink.startRecording(1, 1, 1, 1)
        pylink.beginRealTimeMode(self.params["wait_time"])
        # wait for link data to arrive
        try:
            self.eyelink.waitForBlockStart(self.params["wait_time"], 1, 1)
            return self.codex_msg("rec", "init")
        except RuntimeError as err:
            self.codex_msg("rec", "fail")
            return err

    def end_realtime(self):
        """
        End realtime mode
        """
        pylink.endRealTimeMode()
        self.delay()

    def check_eye(self):
        """
        Check if the eye is being tracked

        Args:
            log (bool): Whether to log the message.

        Returns:
            str or None: The error message.
        """
        # Get the eye information
        available = self.eyelink.eyeAvailable()
        conf_intend = self.params["AutoConfig"]["active_eye"]
        gen_intend = self.params["eye"]

        # Check
        if (
            (available == 0 and gen_intend == 0 and conf_intend == "LEFT") or
            (available == 1 and gen_intend == 1 and conf_intend == "RIGHT") or
            (available == 2 and gen_intend == 2 and conf_intend == "BINOCULAR")
        ):
            self.eyelink.sendMessage(f"EYE_USED {available} {conf_intend}")
            msg = self.codex_msg("eye", "good")
        else:
            msg = self.codex_msg("eye", "bad")

        return msg

    def calibrate(self):
        """
        Calibrate the Eyelink 1000

        Returns:
            str or None: The error message.
        """
        try:
            # Start the calibration
            self.codex_msg("calib", "init")
            self.eyelink.doTrackerSetup()
            res = self.eyelink.getCalibrationMessage()
            if res[-1] == 0:
                return self.codex_msg("calib", "ok")
            else:
                self.codex_msg("calib", "bad")
                return res
        except RuntimeError as err:
            self.eyelink.exitCalibration()
            self.codex_msg("calib", "fail")
            return err

    def terminate(self, host_file, display_file):
        """
        Close the connection to the Eyelink tracker

        Args:
            host_file (str): The file on the host.
            display_file (str): The file on the display.

        Returns:
            str or None: The error message.
        """
        # Check connection
        if not self.eyelink.isConnected():
            return self.codex_msg("con", "fail")
        else:
            # Send the tracker to offline mode
            self.go_offline()
            # Close the edf data file on the Host
            self.eyelink.closeDataFile()
            self.delay()
            # Download the EDF data file from the Host PC to the Display PC
            try:
                self.eyelink.receiveDataFile(str(host_file), str(display_file))
                msg = self.codex_msg("file", "ok")
                pylink.closeGraphics()
                self.eyelink.close()
                return msg
            except RuntimeError as err:
                self.codex_msg("file", "fail")
                pylink.eyelink.closeGraphics()
                self.eyelink.close()
                return err

    def open_edf_file(self, host_file):
        """
        Open a file to record the data and initialize it

        Args:
            host_file (str): The file on the host.

        Returns:
            str or None: The error message.
        """
        error = self.eyelink.openDataFile(host_file)  # Gives 0 or error code
        if not error:
            pretext = f"add_file_preamble_text {self.params['preamble_text']}"
            self.eyelink.sendCommand(pretext)
            self.delay()
            return self.codex_msg("edf", "init")
        else:
            self.codex_msg("edf", "fail")
            return error

    def get_lag(self):
        """
        Get the lag of the Eyelink tracker
        """
        return self.eyelink.trackerTimeUsecOffset()

    def get_time(self):
        """
        Get the time from the Eyelink tracker

        Returns:
            float: The time.
        """
        return self.eyelink.trackerTime()

    def reset(self):
        """
        Flush the Eyelink buffer
        """
        self.eyelink.flushKeybuttons(1)
        pylink.flushGetkeyQueue()
        self.eyelink.resetData()
        self.delay()
        self.codex_msg("reset", "done")

    def drift_correct(self, fix_pos=None):
        """
        Run drift correction

        Args:
            fix (tuple): The fixation point.

        Returns:
            str or None: The error message.
        """
        # Coordinates of the fixation point
        if fix_pos is None:
            monitor = pylink.getDisplayInformation()
            fx, fy = monitor.width / 2, monitor.height / 2
        else:
            fx, fy = fix_pos
        fx, fy = int(fx), int(fy)
        # Perform drift correction
        while True:
            # Checks
            if not self.eyelink.isConnected():
                return self.codex_msg("con", "lost")
            if self.eyelink.breakPressed():
                return self.codex_msg("drift", "term")

            # Perform drift correction
            try:
                err = self.eyelink.doDriftCorrect(fx, fy, 1, 0)
                # break if successful
                if err != pylink.ESC_KEY:
                    # self.eyelink.applyDriftCorrect()
                    break
            except RuntimeError as err:
                self.codex_msg("drift", "fail")
                return err
            
        return self.codex_msg("drift", "ok")

    def check_sample_rate(self):
        """
        Check the sample rate of the Eyelink tracker

        Returns:
            str or None: The error message.
        """
        tracker_rate = self.eyelink.getSampleRate()
        intended_rate = self.params["sample_rate"]
        if tracker_rate == intended_rate:
            status = self.codex_msg("rate", "good")
        else:
            status = self.codex_msg("rate", "bad")
        return status

    def get_correct_eye_data(self, sample):
        """
        Get the correct eye data from a sample

        Args:
            sample (pylink.Sample): The sample data.

        Returns:
            pylink.EyeData or None: The eye data.
        """
        if self.params["eye"] == 0 and sample.isLeftSample():
            data = sample.getLeftEye()
        elif self.params["eye"] == 1 and sample.isRightSample():
            data = sample.getRightEye()
        else:
            data = None
        return data

    def process_samples_online(self):
        """
        Process samples for each event in the link buffer (typically called at the end of each trial)

        Returns:
            list: A list of dictionaries containing the processed sample data.
        """
        # Initialize the information
        info = []
        prev_sample = None

        # Go through the samples in the buffer
        while True:

            # Checks
            if not self.eyelink.isConnected():
                return self.codex_msg("con", "lost", delay=False)
            if self.eyelink.breakPressed():
                return self.codex_msg("con", "term", delay=False)

            # Read the buffer
            item_type = self.eyelink.getNextData()
            if not item_type:
                break

            # Process the sample data
            if item_type == pylink.SAMPLE_TYPE:
                sample = self.eyelink.getFloatData()
                if prev_sample is None or sample.getTime() != prev_sample.getTime():
                    prev_sample = sample
                    info = self.process_sample(sample)

        return info

    def process_sample(self, sample, event_type):
        """
        Process the fixation start event

        Args:
            sample (pylink.Sample): The sample data.
            event_type (int): The event type.

        Returns:
            dict: The processed sample data.
        """
        # Process the general data
        time = sample.getTime()
        ppd_x, ppd_y = sample.getPPD()

        # Process the eye data
        eye_data = self.get_correct_eye_data(sample)
        if eye_data is not None:
            gaze_x, gaze_y = eye_data.getGaze()
            pupil_size = eye_data.getPupilSize()

        return {
            "event_type": event_type,
            "time": time,
            "gaze_x": gaze_x,
            "gaze_y": gaze_y,
            "ppd_x": ppd_x,
            "ppd_y": ppd_y,
            "pupil_size": pupil_size
        }

    def process_events_online(self, events_of_interest: dict = None, process_samples: bool = False):
        """
        Find an event in the Eyelink tracker

        Args:
            events_of_interest (dict): Events to monitor.
            process_samples (bool): Whether to process the samples.

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
        events = []
        samples = []
        prev_sample = None

        # Go through the samples and events in the buffer
        while True:
            # Check
            if not self.eyelink.isConnected():
                return self.codex_msg("con", "lost", delay=False)
            if self.eyelink.breakPressed():
                return self.codex_msg("con", "term", delay=False)

            # Get the event type
            item_type = self.eyelink.getNextData()
            if not item_type:
                break

            # Get the event data for the tracked eye
            if item_type is not None:
                if item_type in events_of_interest.values():
                    ev_data = self.eyelink.getFloatData()
                    if ev_data.getEye() == self.params["eye"]:
                        if item_type == pylink.STARTFIX:
                            func = self.detect_fixation_start_event
                        elif item_type == pylink.ENDFIX:
                            func = self.detect_fixation_update_event
                        elif item_type == pylink.FIXUPDATE:
                            func = self.detect_saccade_start_event
                        elif item_type == pylink.STARTSACC:
                            func = self.process_fixation_end_event
                        elif item_type == pylink.ENDSACC:
                            func = self.process_saccade_end_event
                        # Detect/process the event data
                        events.append(func(ev_data))

                    # Process the sample data for this event
                    if process_samples:
                        while True:
                            # break if no more samples
                            sample = self.eyelink.getNextData()
                            if not sample:
                                break
                            # break if found next event
                            if sample is not None:
                                if sample != pylink.SAMPLE_TYPE:
                                    break
                            if sample == pylink.SAMPLE_TYPE:
                                sam_data = self.eyelink.getFloatData()
                                if prev_sample is None or sam_data.getTime() != prev_sample.getTime():
                                    prev_sample = sam_data
                                    samples.append(self.process_sample(sam_data, item_type))
        if process_samples:
            return events, samples
        else:
            return events

    def detect_fixation_start_event(self, data):
        """
        Process the fixation start event
        """
        time = data.getStartTime()
        gaze_x, gaze_y = data.getStartGaze()
        ppd_x, ppd_y = data.getStartPPD()

        return {
            "event_type": "fixation_start",
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
            "event_type": "fixation_end",
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
            "event_type": "fixation_update",
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
            "event_type": "saccade_start",
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
            "event_type": "saccade_end",
            "time_end": time,
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

        pupil_start = event_data.getStartPupilSize()
        pupil_end = event_data.getEndPupilSize()
        pupil_avg = event_data.getAveragePupilSize()

        return {
            "event_type": "fixation_end",
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

        amp_x, amp_y = event_data.getAmplitude()
        angle = event_data.getAngle()

        velocity_start = event_data.getStartVelocity()
        velocity_end = event_data.getEndVelocity()
        velocity_avg = event_data.getAverageVelocity()
        velocity_peak = event_data.getPeakVelocity()

        return {
            "event_type": "saccade_end",
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
            "amp_x": amp_x,
            "amp_y": amp_y,
            "angle": angle,
            "velocity_start": velocity_start,
            "velocity_end": velocity_end,
            "velocity_avg": velocity_avg,
            "velocity_peak": velocity_peak
        }

    def realtime_gaze_monitor(self, target_pos, valid_dist, min_gaze_dur, clock):
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

    def realtime_fixation_detection(self, target_pos, valid_dist, prev_sample):
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
                gaze_x, gaze_y = self.get_eye_gaze_pos(self.params["eye"], current_sample)
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

    def realtime_saccade_detection(
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
