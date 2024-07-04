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
#                       SPACE: Dartmouth College, Hanover, NH
#
# ==================================================================================================== #
import numpy as np
import pandas as pd

from .psyphy import PsychoPhysicsExperiment
from daedalus.devices.myelink import MyeLink
from daedalus import utils

from psychopy import core
from psychopy.tools.monitorunittools import deg2pix, pix2deg

import pylink
from .EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy


class EyetrackingExperiment(PsychoPhysicsExperiment):
    """
    Eyetracking class for running experiments.

    Args:
        root (str or Path): The root directory of the project.
        platform (str): The platform where the experiment is running.
        debug (bool): Whether to run the experiment in debug mode or not.
    """
    def __init__(self, name, version, root, task_name, platform, mode):

        # Setup
        super().__init__(name, version, root, task_name, platform, mode)

        self.exp_type = "eyetracking"

        # Tracker
        self.tracker = None
        self.genv = None

    def init_tracker(self):
        """
        Initialize the eye tracker.
        """
        # Connect to the eye tracker
        self.tracker = MyeLink(self.name, self.settings.tracker, self.debug)
        self.tracker.go_offline()
        self.tracker.configure(display_name=self.platform)

        # Open graphics environment
        self.tracker.go_offline()
        self.genv = self.make_graphics_env()
        w, h = self.display.window.size
        self.tracker.set_calibration_graphics(w, h, self.genv)

        # Log the initialization
        self.tracker.go_offline()
        self.logger.info("Eyetracker is locked and loaded and ready to go.")
        self.tracker.send_cmd("record_status_message 'Eyetracker configed'")
        self.tracker.codex_msg("tracker", "init")

    def prepare_block(self, block):
        """
        Prepares the block for the experiment.
        """
        self.block_id = self._fix_id(block.id)

        # Make sure tracker is not breaking
        self.tracker.eyelink.terminalBreak(0)

        # Take the tracker offline
        self.tracker.go_offline()

        # Show block info as text
        if block.repeat:
            msg = f"You are repeating block {self.block_id}/{self.task.n_blocks:02d}"
            block.needs_calib = True
            self.logger.info(self.codex.message("block", "rep"))
        else:
            # practice block
            if block.name == "practice":
                block.needs_calib = False
                msg = "You are about to begin a practice block."
            else:
                msg = f"You are about to begin block {self.block_id}/{self.task.n_blocks:02d}"

        # Calibrate if needed
        if block.needs_calib:
            calib_res = self.run_calibration(msg)
            if calib_res == self.codex.message("calib", "term"):
                self.show_msg("Skipping calibration.", msg_type="warning")
            elif calib_res == self.codex.message("calib", "fail"):
                resp = self.show_msg("Calibration failed. Retry (Space) / Continue (Enter)?", msg_type="warning")
                if resp == "space":
                    self.prepare_block(block)
                elif resp == "return":
                    self.logger.warning(calib_res)
                    msg_ = ["Calibration is done."]
                    msg_.append(msg)
                    msg_.append("Press Space to start the block")
                    self.show_msg("\n\n".join(msg_))
                else:
                    self.goodbye(calib_res)
            else:
                msg_ = ["Calibration is done."]
                msg_.append(msg)
                msg_.append("Press Space to start the block")
                self.show_msg("\n\n".join(msg_))
        else:
            msg_ = [msg, "Press Space to start the block"]
            self.show_msg("\n\n".join(msg_))

        # Initialize the block data
        errors = self.files.add_block(self.sub_id, self.ses_id, self.block_id, self.task_name)
        if errors:
            for e in errors:
                self.block_warning(e)

        # Data
        self.data.init_stimuli()
        self.data.init_behavior()
        self.data.init_frames()
        self.data.init_eye_events()
        self.data.init_eye_samples()
        if not self.debug:
            self.open_edf_file()

        # Reset the block clock
        self.timer.start_block()

        # Reset the block status
        if block.repeat:
            block.repeated()

        # Log the start of the block
        self.block_info(self.codex.message("block", "init"))
        self.block_info(f"BLOCKID_{self.block_id}")
        self.tracker.direct_msg(f"BLOCKID {self.block_id}")
        self.tracker.send_cmd(f"record_status_message 'Block {self.block_id}/{self.task.n_blocks:02d}.'")

    def prepare_trial(self, trial):
        """
        Prepares a trial for the experiment by running drift correction and starting the recording.

        Args:
            trial_id (int): The ID of the trial.

        Returns:
            bool: Whether to recalibrate or not.
        """
        self.trial_id = self._fix_id(trial.id)
        self.tracker.eyelink.terminalBreak(0)

        # Establish fixation
        if self.debug:
            recalib = False
            self.timer.start_trial()
        else:
            # Drift correction
            fix_pos = self.cart2mat(*self.stimuli.fixation.pos)
            drift_status = self.tracker.drift_correct(fix_pos)
            if drift_status == self.codex.message("drift", "ok"):
                self.trial_info(drift_status)
                recalib = False

                # Start recording
                on_status = self.tracker.come_online()
                if on_status == self.codex.message("rec", "init"):
                    self.trial_info(on_status)
                    self.timer.start_trial()
                    self.trial_info(f"TRIALID_{self.trial_id}")
                    self.tracker.direct_msg(f"TRIALID {self.trial_id}")
                    self.tracker.send_cmd(f"record_status_message 'Block {self.block_id}, Trial {self.trial_id}.'")
                    self.tracker.direct_msg("!V CLEAR 128 128 128")
                else:
                    self.handle_not_recording(on_status)

            elif drift_status == self.codex.message("drift", "term"):
                self.trial_error(drift_status)
                recalib = True

            elif drift_status == self.codex.message("con", "lost"):
                self.handle_connection_loss(drift_status)
                recalib = self.prepare_trial(trial)

            else:
                recalib = self.handle_drift_error(drift_status)

        # Reset the trial status
        if trial.repeat:
            trial.repeated()

        return recalib

    def handle_drift_error(self, error):
        """
        Handle drift correction errors.

        Args:
            error (str): The error message or code.

        Returns:
            bool: Whether to recalibrate or not.
        """
        self.trial_error(self.codex.message("drift", "fail"))
        self.trial_error(error)
        msg = f"Drift correction failed: {error}\n\nRetry? (Space) / Continue? (Enter)"
        resp = self.show_msg(msg, msg_type="error")
        if resp == "space":
            msg = self.tracker.codex_msg("trial", "rep")
            self.trial_warning(msg)
            recalib = self.prepare_trial()
        else:
            recalib = False
        return recalib

    def handle_not_recording(self, error):
        """
        Handle the case when the eyetracker is not recording.

        Args:
            error (str): The error message or code.
        """
        self.logger.error(self.codex.message("rec", "fail"))
        self.logger.error(error)
        msg = f"Eyetracker is not recording: {error}\n\nReconnect? (Space) / Quit? (Escape)"
        resp = self.show_msg(msg, msg_type="error")
        if resp == "space":
            msg = self.tracker.codex_msg("con", "rep")
            self.logger.warning(msg)
            self.init_tracker()
        else:
            self.goodbye(self.codex.message("usr", "term"))

    def handle_connection_loss(self, error=None):
        """
        Handle the case when the connection to the eyetracker is lost.

        Args:
            error (str): The error message or code.
        """
        msg = "Connection to the eye tracker is lost.\n"
        if error is not None:
            msg += f"{error}\n"
            self.logger.cirtical(error)
        msg += "Reconnect? (Space)"
        self.logger.critical(self.codex.message("con", "lost"))
        resp = self.show_msg(msg, msg_type="error")
        if resp == "space":
            pylink.closeGraphics()
            self.init_tracker()

    def wrap_trial(self):
        """
        Finish up a trial.
        """
        super().wrap_trial()

        # Messages
        self.tracker.codex_msg("trial", "fin")
        self.tracker.direct_msg(f"TRIAL_RESULT {pylink.TRIAL_OK}")
        self.tracker.direct_msg("!V CLEAR 128 128 128")

        # Stop recording
        self.tracker.end_realtime()
        self.tracker.go_offline()
        self.tracker.reset()

    def stop_trial(self):
        """
        Stop a trial midway.
        """
        self.tracker.codex_msg("trial", "stop")
        self.tracker.direct_msg(f"TRIAL_RESULT {pylink.REPEAT_TRIAL}")
        self.tracker.eyelink.terminalBreak(1)
        self.tracker.end_realtime()
        self.tracker.go_offline()
        self.tracker.reset()
        super().stop_trial()

    def wrap_block(self, block):
        """
        Wraps up a block by saving the data and stopping the recording.
        """
        # Stop recording
        self.tracker.codex_msg("block", "fin")
        self.tracker.go_offline()
        self.tracker.reset()
        self.data.save_eye_events(self.files.eye_events)
        self.data.save_eye_samples(self.files.eye_samples)

        # Call the parent method
        super().wrap_block(block)

    def stop_block(self, block):
        """
        Stop the block.
        """
        # Stop recording
        self.tracker.codex_msg("block", "stop")
        self.tracker.eyelink.terminalBreak(1)
        self.tracker.go_offline()
        self.tracker.reset()
        super().stop_block(block)

    def turn_off(self):
        """
        Turn off the experiment.
        """
        self.tracker.send_cmd("record_status_message 'Session is over.'")
        self.tracker.codex_msg("ses", "fin")
        # Close the eye tracker
        status = self.tracker.terminate(self.files.edf_host, self.files.edf_display)
        if status == self.codex.message("file", "ok"):
            self.logger.info("Eye tracker file closed.")
        else:
            # Log
            self.logger.error(status)

        self.logger.info(self.codex.message("ses", "fin"))
        msg = f"You did it. Session {self.ses_id} of the experiment is over."
        msg += "\n\nThank you!"
        self.show_msg(msg, wait_time=self.settings.stimuli["Message"]["duration"], msg_type="info")
        # Save
        self.data.participants["Completed"] = pd.to_datetime(self.data.participants["Completed"])
        self.data.update_participant("Completed", self.timer.today)
        self.data.save_participants(self.files.participants)
        # Quit
        self.goodbye()

    def run_calibration(self, msg=None):
        """
        Run the calibration for the eyetracker.

        Args:
            msg (str): The message to show before calibration.

        Returns:
            str: The status of the calibration.
        """
        # Message
        if msg is None:
            txt = []
        else:
            txt = [msg]
        txt.append("In the next screen, press C to calibrate the eyetracker.")
        txt.append("After calibration press Enter to accept the new calibration and then O to resume the experiment.")
        txt.append("Press Space to continue.")
        resp = self.show_msg("\n\n".join(txt))

        # Run calibration
        self.tracker.send_cmd("record_status_message 'In calibration'")
        if resp == "escape":
            status = self.tracker.codex_msg("calib", "term")
            self.logger.warning(status)
        elif resp == "space":
            self.logger.info(self.codex.message("calib", "init"))
            # Take the tracker offline
            self.tracker.go_offline()
            # Start calibration
            if self.simulation:
                res = self.codex.message("calib", "ok")
            else:
                res = self.tracker.calibrate()
            if res == self.codex.message("calib", "ok"):
                # txt = "Calibration is done. Press Enter to accept the calibration and resume the experiment."
                # resp = self.show_msg(txt)
                # if resp == "return":
                #     self.genv.exit_cal_display()
                #     self.tracker.send_cmd("record_status_message 'Calibration done.'")
                self.logger.info(res)
            else:
                self.logger.error(res)
            status = res

        return status

    def tracker_check(self):
        """
        Runs a check on the eye tracker in the following order:
            1. Start recording
            2. Check the eye
            3. Check the samples

        Returns:
            list: The results of the check.
        """
        # Start testing
        results = []

        # Start recording
        on_status = self.tracker.come_online()
        if on_status == self.codex.message("rec", "init"):
            results.append("(✓) Eye tracker is recording.")
            self.logger.info("Eye tracker is recording.")
        else:
            results.append("(✗) Eye tracker is not recording.")
            self.logger.error("Eye tracker is not recording.")
            self.logger.error(on_status)

        # Check the eye
        eye_status = self.tracker.check_eye()
        if eye_status == self.codex.message("eye", "good"):
            results.append("(✓) Eye checks out.")
            self.logger.info("Eye checks out.")
        else:
            results.append("(✗) Eye mismatch.")
            self.logger.error("Eye mismatch.")

        # Sample rate
        sample_status = self.tracker.check_sample_rate()
        if sample_status == self.codex.message("rate", "good"):
            results.append("(✓) Sample rate checks out.")
            self.logger.info("Sample rate checks out.")
        else:
            intend = self.settings.tracker["sample_rate"]
            rate = self.tracker.eyelink.getSampleRate()
            results.append(f"(✗) Sample rate does not match {intend} vs. {rate}")
            self.logger.error("Sample rate does not match")

        # Set the tracker back to idle mode
        self.tracker.go_offline()

        return results

    def make_graphics_env(self):
        """
        Sets up the graphics environment for the calibration.

        Returns:
            EyeLinkCoreGraphicsPsychoPy: The graphics environment for the calibration.
        """
        # Window
        # Eye tracking experiments always use pixel measurements.
        self.display.window.units = 'pix'
        # NOTE: If we don't do this early, the eyetracker will override it and we always get a grey background.
        self.display.window.color = utils.str2tuple(self.settings.stimuli["Display"]["background_color"])

        # Configure a graphics environment (genv) for tracker calibration
        genv = EyeLinkCoreGraphicsPsychoPy(self.tracker.eyelink, self.display.window)

        # Set background and foreground colors for the calibration target
        foreground_color = utils.str2tuple(self.settings.tracker["Calibration"]["target_color"])
        background_color = utils.str2tuple(self.settings.tracker["Calibration"]["background_color"])
        genv.setCalibrationColors(foreground_color, background_color)

        # Set up the calibration target
        genv.setTargetType('circle')
        genv.setTargetSize(deg2pix(self.settings.tracker["Calibration"]["target_size"], self.display.monitor))

        # Set up the calibration sounds
        genv.setCalibrationSounds("", "", "")

        return genv

    @staticmethod
    def gaze_in_square(gaze, center, length):
        """
        Check if the gaze is within a target square.

        Args:
            gaze (tuple): The gaze coordinates.
            center (tuple): The center of the square.
            length (float): The length of the square.

        Returns:
            bool: Whether the fixation is within the square or not.
        """
        gx, gy = gaze
        cx, cy = center
        valid_x = cx - length / 2 < gx < cx + length / 2
        valid_y = cy - length / 2 < gy < cy + length / 2

        if valid_x and valid_y:
            return True
        else:
            return False

    @staticmethod
    def gaze_in_circle(gaze, center, radius):
        """
        Check if the gaze is within a target circle.

        Args:
            gaze (tuple): The gaze coordinates.
            center (tuple): The center of the circle.
            radius (float): The radius of the circle.

        Returns:
            bool: Whether the fixation is within the circle or not.
        """
        offset = utils.get_hypotenus(gaze, center)
        if offset < radius:
            return True
        else:
            return False

    def mat2cart(self, x, y):
        """
        Center the Eyelink coordinates.

        Args:
            x (float): The x-coordinate.
            y (float): The y-coordinate.

        Returns:
            tuple: The centered coordinates.
        """
        return x - self.display.window.size[0] / 2, self.display.window.size[1] / 2 - y

    def cart2mat(self, x, y):
        """
        Convert center coordinates to matrix coordinates.

        Args:
            x (float): The x-coordinate.
            y (float): The y-coordinate.

        Returns:
            tuple: The matrix coordinates.
        """
        return self.display.window.size[0] / 2 + x, self.display.window.size[1] / 2 - y

    def establish_fixation(self, method="circle", fix=None):
        """
        Establish fixation.

        Args:
            method (str): The method to use for fixation.
            fix (object): The fixation stimulus.

        Returns:
            bool: Whether the fixation is established or not.
        """
        # Setup
        fix_event = {"fixation_start": pylink.STARTFIX}
        fix_radi = deg2pix(self.settings.exp["General"]["valid_fixation_radius"], self.display.monitor)
        if fix is None:
            fix = self.stimuli.fixation
        tout = self.settings.exp["General"]["fixation_timeout"]

        # Look for fixation
        start_time = self.timer.trial.getTime()
        while True:

            if self.timer.trial.getTime() - start_time > tout:
                msg = self.tracker.codex_msg("fix", "timeout")
                self.trial_error(msg)
                return msg

            # Draw fixation
            self.display.show(fix)

            # Check eye events
            events = self.tracker.process_events_online(fix_event)
            if events == self.codex.message("con", "lost"):
                self.trial_error(events)
                return msg

            for event in events:
                # get the gaze data
                gaze_x = event["gaze_start_x"]
                gaze_y = event["gaze_start_y"]

                # Check if the gaze is within the fixation
                gx, gy = self.mat2cart(gaze_x, gaze_y)
                if method == "circle":
                    valid = self.gaze_in_circle((gx, gy), fix.pos, fix_radi)
                else:
                    valid = self.gaze_in_square((gx, gy), fix.pos, fix_radi)
                if valid:
                    msg = self.tracker.codex_msg("fix", "ok")
                    self.trial_info(msg)
                    return msg

    def monitor_fixation(self, fix_pos=None, method="circle"):
        """
        Monitor fixation.

        Args:
            fixation (object): The fixation stimulus.
            timeout (float): The time to wait for fixation.
            radi (float): The radius of the circle.

        Returns:
            bool: Whether the fixation is established or not.
        """
        # Setup
        status = None
        fix_radi = deg2pix(self.settings.exp["General"]["valid_fixation_radius"], self.display.monitor)
        if fix_pos is None:
            fix_pos = self.stimuli.fixation.pos

        # Get events
        fix_events = {"fixation_update": pylink.FIXUPDATE, "fixation_end": pylink.ENDFIX}
        events = self.tracker.process_events_online(fix_events)
        # Check if there was an error
        if events in [self.codex.message("con", "lost"), self.codex.message("con", "term")]:
            self.trial_error(events)
            return status, events

        # Process the events
        if events:
            for event in events:
                # get gaze data
                gaze_x = event["gaze_avg_x"]
                gaze_y = event["gaze_avg_y"]
                gx, gy = self.mat2cart(gaze_x, gaze_y)
                # check if gaze is within the fixation
                if method == "circle":
                    valid = self.gaze_in_circle((gx, gy), fix_pos, fix_radi)
                else:
                    valid = self.gaze_in_square((gx, gy), fix_pos, fix_radi)
                # check if fixation is valid
                if not valid:
                    status = self.codex.message("fix", "bad")
                    self.trial_error(status)
                    self.tracker.direct_msg(status, delay=False)
                    break
                else:
                    status = self.codex.message("fix", "ok")
        else:
            status = self.codex.message("fix", "null")

        return status, events

    def find_saccade(self, target_positions, fix_pos=None, fix_method="circle"):
        """
        Look for saccade.

        Args:
            target_positions (list): The position of the target.
            fix_pos (tuple): The position of the fixation. If None, the position of fixation stimulus is used.
            fix_method (str): The method to use for fixation.

        Returns:
            bool: Whether the fixation is established or not.
        """
        # Setup
        status = None
        fix_radi = deg2pix(self.settings.exp["General"]["valid_fixation_radius"], self.display.monitor)
        sacc_radi = deg2pix(self.settings.exp["General"]["valid_saccade_radius"], self.display.monitor)
        if fix_pos is None:
            fix_pos = self.stimuli.fixation.pos

        # Get events
        mon_events = {
            "fixation_update": pylink.FIXUPDATE,
            "fixation_end": pylink.ENDFIX,
            "saccade_start": pylink.STARTSACC,
            "saccade_end": pylink.ENDSACC
        }
        events = self.tracker.process_events_online(mon_events)
        # Check if there was an error
        if events in [self.codex.message("con", "lost"), self.codex.message("con", "term")]:
            self.trial_error(events)
            return status, events

        # Find event types that are found
        if events:
            # Process events
            # If there is a saccade end, then it should override fixation update
            # Because it has happened after fixating on the fixation point, but we might get
            # updates after the saccade end too when the target is fixated (?). We don't want
            # to process the validity of those updates.
            # Fixation end and saccade start are not processed here and are saved for later use
            for event in events:
                if event["event_type"] == "fixation_update":
                    gaze_x = event["gaze_avg_x"]
                    gaze_y = event["gaze_avg_y"]
                    gx, gy = self.mat2cart(gaze_x, gaze_y)
                    if fix_method == "circle":
                        valid = self.gaze_in_circle((gx, gy), fix_pos, fix_radi)
                    else:
                        valid = self.gaze_in_square((gx, gy), fix_pos, fix_radi)
                    if not valid:
                        status = self.codex.message("fix", "bad")
                        self.trial_error(status)
                        self.tracker.direct_msg(status, delay=False)
                        break
                    else:
                        status = self.codex.message("fix", "ok")
                elif event["event_type"] == "saccade_end":
                    gaze_x = event["gaze_end_x"]
                    gaze_y = event["gaze_end_y"]
                    gx, gy = self.mat2cart(gaze_x, gaze_y)
                    # check if close to target(s)
                    landings = []
                    for tp in target_positions:
                        if fix_method == "circle":
                            valid = self.gaze_in_circle((gx, gy), tp, sacc_radi)
                        else:
                            valid = self.gaze_in_square((gx, gy), tp, sacc_radi)
                        landings.append(valid)
                    if any(landings):
                        status = self.codex.message("sacc", "good")
                    else:
                        status = self.codex.message("sacc", "bad")
                        self.trial_error(status)
                        self.tracker.direct_msg(status, delay=False)
                    break
        else:
            status = self.codex.message("sacc", "null")

        return status, events

    def get_sacc_landing(self, events):
        """
        Get the end gaze coordinates.

        Args:
            event (dict): The event dictionary.

        Returns:
            tuple: The gaze coordinates.
        """
        for event in events:
            if event["event_type"] == "saccade_end":
                gx, gy = self.mat2cart(event["gaze_end_x"], event["gaze_end_y"])
        return gx, gy

    def raw_events_to_df(self, events):
        """
        Convert raw events to a DataFrame without any processing.

        Args:
            events (list): The list of events, each is a dictionary.

        Returns:
            pd.DataFrame: The DataFrame of the events.
        """
        df = pd.DataFrame(events)
        return df

    def parse_fixation_update_event(self, event, trial):
        """
        Parse the fixation update event.

        Args:
            event (dict): The fixation update event.

        Returns:
            tuple: The gaze coordinates.
        """
        start = event["time_start"]
        end = event["time_end"]
        duration = end - start
        start_idx = utils.time_index(start, trial.choice_tracker_times)
        start_frame = trial.choice_frames[start_idx]
        end_idx = utils.time_index(end, trial.choice_tracker_times)
        end_frame = trial.choice_frames[end_idx]
        gaze_avg_x, gaze_avg_y = self.mat2cart(event["gaze_avg_x"], event["gaze_avg_y"])
        ppd_avg_x, ppd_avg_y = self.mat2cart(event["ppd_avg_x"], event["ppd_avg_y"])

        df = pd.DataFrame({
            "EventType": ["FixationUpdate"],
            "EventStart_FrameN": [start_frame],
            "EventStart_TrackerTime_ms": [start],
            "EventStart_TrialTime_ms": [trial.choice_times[start_frame]],
            "EventEnd_FrameN": [end_frame],
            "EventEnd_TrackerTime_ms": [end],
            "EventEnd_TrialTime_ms": [trial.choice_times[end_frame]],
            "EventDuration_ms": [duration],
            "GazeAvgX_px": [gaze_avg_x],
            "GazeAvgY_px": [gaze_avg_y],
            "GazeAvgX_ppd": [ppd_avg_x],
            "GazeAvgY_ppd": [ppd_avg_y],
            "GazeAvgX_dva": [pix2deg(gaze_avg_x, self.display.monitor)],
            "GazeAvgY_dva": [pix2deg(gaze_avg_y, self.display.monitor)],
            "GazeAvgX_Tracker_dva": [gaze_avg_x / ppd_avg_x],
            "GazeAvgY_Tracker_dva": [gaze_avg_y / ppd_avg_y]
        })

        return df

    def parse_fixation_end_event(self, event, trial):
        """
        Parse the fixation end event.

        Args:
            event (dict): The fixation end event.
            trial (Trial): The trial object.

        Returns:
            pd.DataFrame: The parsed data.
        """
        start = event["time_start"]
        end = event["time_end"]
        start_idx = utils.time_index(start, trial.choice_tracker_times)
        start_frame = trial.choice_frames[start_idx]
        end_idx = utils.time_index(end, trial.choice_tracker_times)
        end_frame = trial.choice_frames[end_idx]
        gaze_start_x, gaze_start_y = self.mat2cart(event["gaze_start_x"], event["gaze_start_y"])
        gaze_end_x, gaze_end_y = self.mat2cart(event["gaze_end_x"], event["gaze_end_y"])
        gaze_avg_x, gaze_avg_y = self.mat2cart(event["gaze_avg_x"], event["gaze_avg_y"])
        ppd_start_x, ppd_start_y = self.mat2cart(event["ppd_start_x"], event["ppd_start_y"])
        ppd_end_x, ppd_end_y = self.mat2cart(event["ppd_end_x"], event["ppd_end_y"])
        ppd_avg_x, ppd_avg_y = self.mat2cart(event["ppd_avg_x"], event["ppd_avg_y"])

        df = pd.DataFrame({
            "EventType": ["FixationEnd"],
            "EventDuration_ms": [end - start],
            "EventStart_FrameN": [start_frame],
            "EventStart_TrackerTime_ms": [start],
            "EventStart_TrialTime_ms": [trial.choice_times[start_frame]],
            "GazeStartX_px": [gaze_start_x],
            "GazeStartY_px": [gaze_start_y],
            "GazeStartX_ppd": [ppd_start_x],
            "GazeStartY_ppd": [ppd_start_y],
            "GazeStartX_dva": [pix2deg(gaze_start_x, self.display.monitor)],
            "GazeStartY_dva": [pix2deg(gaze_start_y, self.display.monitor)],
            "GazeStartX_Tracker_dva": [gaze_start_x / ppd_start_x],
            "GazeStartY_Tracker_dva": [gaze_start_y / ppd_start_y],
            "EventEnd_FrameN": [end_frame],
            "EventEnd_TrackerTime_ms": [end],
            "EventEnd_TrialTime_ms": [trial.choice_times[end_frame]],
            "GazeEndX_px": [gaze_end_x],
            "GazeEndY_px": [gaze_end_y],
            "GazeEndX_ppd": [ppd_end_x],
            "GazeEndY_ppd": [ppd_end_y],
            "GazeEndX_dva": [pix2deg(gaze_end_x, self.display.monitor)],
            "GazeEndY_dva": [pix2deg(gaze_end_y, self.display.monitor)],
            "GazeEndX_Tracker_dva": [gaze_end_x / ppd_end_x],
            "GazeEndY_Tracker_dva": [gaze_end_y / ppd_end_y],
            "GazeAvgX_px": [gaze_avg_x],
            "GazeAvgY_px": [gaze_avg_y],
            "GazeAvgX_ppd": [ppd_avg_x],
            "GazeAvgY_ppd": [ppd_avg_y],
            "GazeAvgX_dva": [pix2deg(gaze_avg_x, self.display.monitor)],
            "GazeAvgY_dva": [pix2deg(gaze_avg_y, self.display.monitor)],
            "GazeAvgX_Tracker_dva": [gaze_avg_x / ppd_avg_x],
            "GazeAvgY_Tracker_dva": [gaze_avg_y / ppd_avg_y],
            "PupilStart": [event["pupil_start"]],
            "PupilEnd": [event["pupil_end"]],
            "PupilAvg": [event["pupil_avg"]]
        })

        return df

    def parse_saccade_start_event(self, event, trial):
        """
        Parse the saccade start event.

        Args:
            event (dict): The saccade start event.
            trial (Trial): The trial object.

        Returns:
            pd.DataFrame: The parsed data.
        """
        start = event["time_start"]
        start_idx = utils.time_index(start, trial.choice_tracker_times)
        start_frame = trial.choice_frames[start_idx]
        gaze_start_x, gaze_start_y = self.mat2cart(event["gaze_start_x"], event["gaze_start_y"])
        ppd_start_x, ppd_start_y = self.mat2cart(event["ppd_start_x"], event["ppd_start_y"])

        df = pd.DataFrame({
            "EventType": ["SaccadeStart"],
            "EventStart_FrameN": [start_frame],
            "EventStart_TrackerTime_ms": [start],
            "EventStart_TrialTime_ms": [trial.choice_times[start_frame]],
            "GazeStartX_px": [gaze_start_x],
            "GazeStartY_px": [gaze_start_y],
            "GazeStartX_ppd": [ppd_start_x],
            "GazeStartY_ppd": [ppd_start_y],
            "GazeStartX_dva": [pix2deg(gaze_start_x, self.display.monitor)],
            "GazeStartY_dva": [pix2deg(gaze_start_y, self.display.monitor)],
            "GazeStartX_Tracker_dva": [gaze_start_x / ppd_start_x],
            "GazeStartY_Tracker_dva": [gaze_start_y / ppd_start_y]
        })

        return df

    def parse_saccade_end_event(self, event, trial):
        """
        Parse the saccade end event.

        Args:
            event (dict): The saccade end event.
            trial (Trial): The trial object.

        Returns:
            pd.DataFrame: The parsed data.
        """
        start = event["time_start"]
        end = event["time_end"]
        start_idx = utils.time_index(start, trial.choice_tracker_times)
        start_frame = trial.choice_frames[start_idx]
        end_idx = utils.time_index(end, trial.choice_tracker_times)
        end_frame = trial.choice_frames[end_idx]
        gaze_start_x, gaze_start_y = self.mat2cart(event["gaze_start_x"], event["gaze_start_y"])
        gaze_end_x, gaze_end_y = self.mat2cart(event["gaze_end_x"], event["gaze_end_y"])
        ppd_start_x, ppd_start_y = self.mat2cart(event["ppd_start_x"], event["ppd_start_y"])
        ppd_end_x, ppd_end_y = self.mat2cart(event["ppd_end_x"], event["ppd_end_y"])
        amp_x = event["amp_x"]
        amp_y = event["amp_y"]
        vel_start = event["velocity_start"]
        vel_end = event["velocity_end"]
        vel_peak = event["velocity_peak"]
        vel_avg = event["velocity_avg"]

        df = pd.DataFrame({
            "EventType": ["SaccadeEnd"],
            "EventDuration_ms": [end - start],
            "EventStart_FrameN": [start_frame],
            "EventStart_TrackerTime_ms": [start],
            "EventStart_TrialTime_ms": [trial.choice_times[start_frame]],
            "GazeStartX_px": [gaze_start_x],
            "GazeStartY_px": [gaze_start_y],
            "GazeStartX_ppd": [ppd_start_x],
            "GazeStartY_ppd": [ppd_start_y],
            "GazeStartX_dva": [pix2deg(gaze_start_x, self.display.monitor)],
            "GazeStartY_dva": [pix2deg(gaze_start_y, self.display.monitor)],
            "GazeStartX_Tracker_dva": [gaze_start_x / ppd_start_x],
            "GazeStartY_Tracker_dva": [gaze_start_y / ppd_start_y],
            "EventEnd_FrameN": [end_frame],
            "EventEnd_TrackerTime_ms": [end],
            "EventEnd_TrialTime_ms": [trial.choice_times[end_frame]],
            "GazeEndX_px": [gaze_end_x],
            "GazeEndY_px": [gaze_end_y],
            "GazeEndX_ppd": [ppd_end_x],
            "GazeEndY_ppd": [ppd_end_y],
            "GazeEndX_dva": [pix2deg(gaze_end_x, self.display.monitor)],
            "GazeEndY_dva": [pix2deg(gaze_end_y, self.display.monitor)],
            "GazeEndX_Tracker_dva": [gaze_end_x / ppd_end_x],
            "GazeEndY_Tracker_dva": [gaze_end_y / ppd_end_y],
            "SaccAmpX_dva": [amp_x],
            "SaccAmpX_px": [deg2pix(amp_x, self.display.monitor)],
            "SaccAmpY_dva": [amp_y],
            "SaccAmpY_px": [deg2pix(amp_y, self.display.monitor)],
            "SaccStartVel_dps": [vel_start],
            "SaccEndVel_dps": [vel_end],
            "SaccStartVel_pps": [self.dps2pps(vel_start)],
            "SaccEndVel_pps": [self.dps2pps(vel_end)],
            "SaccPeakVel_dps": [event['velocity_peak']],
            "SaccPeakVel_pps": [self.dps2pps(vel_peak)],
            "SaccAvgVel_dps": [event['velocity_avg']],
            "SaccAvgVel_pps": [self.dps2pps(vel_avg)],
            "SaccAngle_deg": [event['angle']],
            "SaccAngle_rad": [np.radians(event['angle'])]
        })

        return df


    def add_to_events_dataframe(self, data: dict, tracker_lag: float):
        """
        Add a dictionary of data to the events dataframe.

        Args:
            data (dict): The data to add.
            tracker_lag (float): The lag of the tracker.

        Returns:
            pd.DataFrame: The updated dataframe.
        """
        df = pd.DataFrame()
        df["BlockID"] = self.block_id
        df["BlockName"] = self.block.name
        df["TrialNumber"] = self.trial_id
        tidx = int(self.trial_id) - 1
        df["TrialIndex"] = tidx
        df["TrackerLag"] = tracker_lag
        df["EventType"] = data["event_type"]
        # time
        ts = data.get("time_start")
        if ts is not None:
            df["EventStart_TrackerTime_ms"] = ts
            df["EventStart_TrialTime_ms"] = ts - tracker_lag
            frame_n = utils.time_index_from_sum(ts, self.window.frameIntervals)
            df["EventStart_FrameN"] = frame_n

            vec_coder = np.vectorize(self.codex.get_proc_name)
            df["EventStart_Period"] = vec_coder(self.task.block.trial.data[frame_n])
        te = data.get("time_end")
        if te is not None:
            df["EventEnd_TrackerTime_ms"] = te
            df["EventEnd_TrialTime_ms"] = te - tracker_lag
            frame_n = utils.time_index_from_sum(te, self.window.frameIntervals)
            df["EventEnd_FrameN"] = frame_n
            df["EventEnd_Period"] = vec_coder(self.task.block.trial.data[frame_n])
        dur = data.get("duration")
        if dur is not None:
            df["EventDuration_ms"] = dur
            df["EventDuration_fr"] = self.ms2fr(dur)
        # start gaze
        gsx = data.get("gaze_start_x")
        gsy = data.get("gaze_start_y")
        if gsx is not None and gsy is not None:
            gsx, gsy = self.mat2cart(gsx, gsy)
            df["GazeStartX_px"] = gsx
            df["GazeStartY_px"] = gsy
            ppdsx = data.get("ppd_start_x")
            ppdsy = data.get("ppd_start_y")
            if ppdsx is not None and ppdsy is not None:
                df["GazeStartX_ppd"] = ppdsx
                df["GazeStartY_ppd"] = ppdsy
                gsx_dva = gsx / ppdsx
                gsy_dva = gsy / ppdsy
                df["GazeStartX_dva"] = gsx_dva
                df["GazeStartY_dva"] = gsy_dva
        # end gaze
        gex = data.get("gaze_end_x")
        gey = data.get("gaze_end_y")
        if gex is not None and gey is not None:
            gex, gey = self.mat2cart(gex, gey)
            df["GazeEndX_px"] = gex
            df["GazeEndY_px"] = gey
            ppdex = data.get("ppd_end_x")
            ppdey = data.get("ppd_end_y")
            if ppdex is not None and ppdey is not None:
                df["GazeEndX_ppd"] = ppdex
                df["GazeEndY_ppd"] = ppdey
                gex = gex / ppdex
                gey = gey / ppdey
                df["GazeEndX_dva"] = gex
                df["GazeEndY_dva"] = gey
        # average gaze
        gavgx = data.get("gaze_avg_x")
        gavgy = data.get("gaze_avg_y")
        if gavgx is not None and gavgy is not None:
            gavgx, gavgy = self.mat2cart(gavgx, gavgy)
            df["GazeAvgX_px"] = gavgx
            df["GazeAvgY_px"] = gavgy
            ppdavgx = data.get("ppd_avg_x")
            ppdavgy = data.get("ppd_avg_y")
            if ppdavgx is not None and ppdavgy is not None:
                df["GazeAvgX_ppd"] = ppdavgx
                df["GazeAvgY_ppd"] = ppdavgy
                gavgx_dva = gavgx / ppdavgx
                gavgy_dva = gavgy / ppdavgy
                df["GazeAvgX_dva"] = gavgx_dva
                df["GazeAvgY_dva"] = gavgy_dva
        # amplitude
        ampx = data.get("amp_x")
        ampy = data.get("amp_y")
        if ampx is not None and ampy is not None:
            df["AmplitudeX_dva"] = ampx
            df["AmplitudeY_dva"] = ampy
        # angle
        ang = data.get("angle")
        if ang is not None:
            df["Angle_deg"] = ang
            df["Angle_rad"] = np.radians(ang)
        # velocity
        sv = data.get("velocity_start")
        if sv is not None:
            df["VelocityStart_dps"] = sv
        ev = data.get("velocity_end")
        if ev is not None:
            df["VelocityEnd_dps"] = ev
        avgv = data.get("velocity_avg")
        if avgv is not None:
            df["VelocityAvg_dps"] = avgv
        pv = data.get("velocity_peak")
        if pv is not None:
            df["VelocityPeak_dps"] = pv
        # pupil
        ps = data.get("pupil_start")
        if ps is not None:
            df["PupilStart_area"] = ps
        pe = data.get("pupil_end")
        if pe is not None:
            df["PupilEnd_area"] = pe
        avgp = data.get("pupil_avg")
        if avgp is not None:
            df["PupilAvg_area"] = avgp

        # Concatenate the data
        return df

    def open_edf_file(self):
        """
        """
        # Display file
        status = self.tracker.open_edf_file(self.files.edf_host)
        if status == self.codex.message("edf", "init"):
            self.block_info(status)
        else:
            self.goodbye(status)

    def goodbye(self, raise_error=None):
        """
        End the experiment. Close the window and the tracker.
        If raising an error (quitting in the middle of the experiment), save as much data as possible.

        Args:
            raise_error (str): The error message to raise.
        """
        # Close the window
        self.display.clear()
        self.display.close()

        # Quit
        if raise_error is not None:
            # Close the eye tracker
            self.tracker.codex_msg("exp", "term")
            self.tracker.eyelink.terminalBreak(1)
            self.tracker.terminate(self.files.edf_host, self.files.edf_display)
            # Log the error
            self.logger.critical(raise_error)
            # Save as much as you can
            try:
                self.data.save_stimuli(self.files.stim_data)
                self.data.save_behavior(self.file.behavior)
                self.data.save_frames(self.files.frames)
                self.data.save_eye_events(self.files.eye_events)
                self.data.save_eye_samples(self.files.eye_samples)
                self.logger.close_file()
            except AttributeError as e:
                print(f"Error saving data: {e}")
                # Raise the error
                raise SystemExit(f"Experiment ended with error: {raise_error}")
        else:
            self.logger.info(f"Bye Bye Experiment. Duration: {self.timer.exp.getTime() / 60:.3f} minutes")
            self.logger.close_file()
            core.quit()

    @staticmethod
    def trial2tracker_time(target_time, tracker_frames, trial_frames):
        """
        Convert trial time to tracker time.

        Args:
            target_time (float): target time
            tracker_frames (np.array): tracker frame times
            trial_frames (np.array): trial frame times

        Returns:
            tuple: frame index, tracker time
        """
        # Find the frame index closest to the target time
        frame_idx = np.argmin(np.abs(trial_frames - target_time))

        # Find the tracker time at that frame
        tracker_time = tracker_frames[frame_idx]

        return frame_idx, tracker_time

    @staticmethod
    def tracker2trial_time(target_time, tracker_frames, trial_frames):
        """
        Convert tracker time to experiment time.

        Args:
            target_time (float): target time
            tracker_frames (np.array): tracker frames
            trial_frames (np.array): trial

        Returns:
            tuple: frame index, trial time
        """
        # Find the closest frame
        frame_idx = np.argmin(np.abs(tracker_frames - target_time))

        # Find the corresponding trial time
        trial_time = trial_frames[frame_idx]

        return frame_idx, trial_time
