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
from psychopy.tools.monitorunittools import deg2pix

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
    def __init__(
        self,
        root,
        platform: str,
        debug: bool,
        tracker_model: str = "Eyelink1000Plus"
    ):
        # Setup
        super().__init__(root, platform, debug)

        self.exp_type = "eyetracking"

        # Tracker
        self.tracker_model = tracker_model
        self.tracker_params = None
        self.tracker = None
        self.genv = None

        # Files
        self.events_data_file = None
        self.samples_data_file = None
        self.edf_display_file = None
        self.edf_host_file = None

        # Data
        self.events_data = None
        self.samples_data = None
        self.edf_display_file = None
        self.edf_host_file = None

    def init_tracker(self):
        """
        Initialize the eye tracker.
        """
        # Connect to the eye tracker
        conf_file = self.root / "config" / "eyetrackers.yaml"
        self.tracker_params = utils.read_config(conf_file)[self.tracker_model]
        self.tracker = MyeLink(self.name, self.tracker_model, self.tracker_params, self.debug)
        self.tracker.go_offline()
        self.tracker.configure(display_name=self.platform)

        # Configure the eye tracker
        self.logger.debug(f"Eyetrakcer mode: {self.tracker.eyelink.getTrackerMode()}")
        self.logger.debug(f"Eyetrakcer version: {self.tracker.eyelink.getTrackerVersion()}")
        self.logger.debug(f"Eyetrakcer info: {self.tracker.eyelink.getTrackerInfo().getAddress()}")
        self.logger.debug(f"Eyetrakcer info: {self.tracker.eyelink.getTrackerInfo().getTrackerName()}")
        self.logger.debug(f"Eyetrakcer info: {self.tracker.eyelink.getTrackerInfo().getEventDataFlags()}")

        # Open graphics environment
        self.tracker.go_offline()
        self.genv = self.make_graphics_env()
        w, h = self.window.size
        self.logger.debug(f"Graphics environment: {self.genv}")
        self.logger.debug(f"Window size: {w}x{h}")
        self.tracker.set_calibration_graphics(w, h, self.genv)

        # Log the initialization
        self.tracker.go_offline()
        self.logger.info("Eyetracker is locked and loaded and ready to go.")
        self.tracker.send_cmd("record_status_message 'Eyetracker configed'")
        self.tracker.codex_msg("tracker", "init")

    def prepare_block(self, block_id, repeat=False, calib=False):
        """
        Prepares the block for the experiment.

        Args:
            block_id (int): The ID of the block.
            repeat (bool): Whether the block is being repeated or not.
            calib (bool): Whether to calibrate or not.
        """
        # Set the ID of the block
        self.block_id = self._fix_id(block_id)

        # Make sure tracker is not breaking
        self.tracker.eyelink.terminalBreak(0)

        # Take the tracker offline
        self.tracker.go_offline()

        # Show block info as text
        n_blocks = f"{len(self.all_blocks):02d}"
        if repeat:
            msg = f"You are repeating block {self.block_id}/{n_blocks}"
            calib = True
            self.logger.info(self.codex.message("block", "rep"))
            self.logger.info(self.codex.message("calib", "rep"))
        else:
            # practice block
            if int(block_id) == 0:
                calib = True
                self.logger.debug("Practice block. Calibrating.")
            msg = f"You are about to begin block {self.block_id}/{n_blocks}"

        # Calibrate if needed
        if calib:
            calib_res = self.run_calibration(msg)
            if calib_res == self.codex.message("calib", "term"):
                self.show_msg("Skipping calibration.", msg_type="warning")
            elif calib_res == self.codex.message("calib", "fail"):
                resp = self.show_msg("Calibration failed. Retry (Space) / Continue (Enter)?", msg_type="warning")
                if resp == "space":
                    self.prepare_block(self.block_id, repeat, calib)
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

        # Reset the block clock
        self.block_clock.reset()
        # Initialize the block data
        self.init_block_data()
        # Log the start of the block
        self.block_info(self.codex.message("block", "init"))
        self.block_info(f"BLOCKID_{self.block_id}")
        self.tracker.direct_msg(f"BLOCKID {self.block_id}")
        self.tracker.send_cmd(f"record_status_message 'Block {self.block_id}/{n_blocks}.'")

    def prepare_trial(self, trial_id):
        """
        Prepares a trial for the experiment by running drift correction and starting the recording.

        Args:
            trial_id (int): The ID of the trial.

        Returns:
            bool: Whether to recalibrate or not.
        """
        self.trial_id = self._fix_id(trial_id)
        self.tracker.eyelink.terminalBreak(0)

        # Establish fixation
        if self.debug:
            recalib = False
            self.trial_clock = core.MonotonicClock()
            self.trial_info(self.codex.message("trial", "init"))
            self.trial_info(f"TRIALID_{self.trial_id}")
        else:
            fix_status = self.establish_fixation()
            if fix_status == self.codex.message("fix", "ok"):
                self.trial_debug("Fixation is established.")
                # Drift correction
                fix_pos = self.cart2mat(*self.fix_stim.pos)
                drift_status = self.tracker.drift_correct(fix_pos)
                if drift_status == self.codex.message("drift", "ok"):
                    self.trial_info(drift_status)
                    recalib = False
                    # Start recording
                    on_status = self.tracker.come_online()
                    if on_status == self.codex.message("rec", "init"):
                        self.trial_info(on_status)
                        self.trial_clock = core.MonotonicClock()
                        self.trial_info(f"TRIALID_{self.trial_id}")
                        self.tracker.direct_msg(f"TRIALID {self.trial_id}")
                        self.tracker.send_cmd(f"record_status_message 'Block {self.block_id}, Trial {self.trial_id}.'")
                        self.tracker.direct_msg("!V CLEAR 128 128 128")
                    else:
                        self.handle_not_recording(on_status)
                elif drift_status == self.codex.message("drift", "term"):
                    self.trial_warning(drift_status)
                    recalib = True
                elif drift_status == self.codex.message("con", "lost"):
                    self.handle_connection_loss(drift_status)
                    recalib = self.prepare_trial()
                else:
                    recalib = self.handle_drift_error(drift_status)
            # Fixation timeout
            elif fix_status == self.codex.message("fix", "timeout"):
                recalib = True
            else:
                self.handle_connection_loss(fix_status)
                recalib = self.prepare_trial()

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
        msg = f"Drift correction failed.\n{error}\nRetry? (Space)"
        resp = self.show_msg(msg, msg_type="error")
        if resp == "space":
            msg = self.tracker.codex_msg("trial", "rep")
            self.trial_warning(msg)
            recalib = self.prepare_trial(self.trial_id)
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
        msg = f"Eyetracker is not recording.\n{error}\nReconnect? (Space)"
        resp = self.show_msg(msg, msg_type="error")
        if resp == "space":
            msg = self.tracker.codex_msg("con", "rep")
            self.logger.warning(msg)
            self.init_tracker()

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

    def wrap_block(self):
        """
        Wraps up a block by saving the data and stopping the recording.
        """
        # Stop recording
        self.tracker.codex_msg("block", "fin")
        self.tracker.go_offline()
        self.tracker.reset()

        # Call the parent method
        super().wrap_block()

        # Save the data
        self.save_events_data()
        self.save_samples_data()

    def stop_block(self):
        """
        Stop the block.
        """
        # Stop recording
        self.tracker.codex_msg("block", "stop")
        self.tracker.eyelink.terminalBreak(1)
        self.tracker.go_offline()
        self.tracker.reset()
        super().stop_block()

    def turn_off(self):
        """
        Turn off the experiment.
        """
        # Log the end of the session
        self.logger.info(self.codex.message("ses", "fin"))
        self.tracker.send_cmd("record_status_message 'Session is over.'")
        self.tracker.codex_msg("ses", "fin")
        self.tracker.terminate(self.edf_host_file, self.edf_display_file)
        super().turn_off()

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
        txt.append("After the procedure, press Enter to accept the new calibration and then O to resume the experiment.")
        txt.append("Press Space to continue.")
        resp = self.show_msg("\n\n".join(txt))

        # Run calibration
        self.tracker.send_cmd("record_status_message 'In calibration'")
        if resp == "escape":
            self.logger.warning(status)
            status = self.tracker.codex_msg("calib", "term")
        elif resp == "space":
            self.logger.info(self.codex.message("calib", "init"))
            # Take the tracker offline
            self.tracker.go_offline()
            # Start calibration
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

        # Check the samples
        # t = 500
        # self.tracker.delay()
        # samples = self.tracker.process_samples_online()
        # if isinstance(samples, list):
        #     if len(samples) > 0:
        #         results.append(f"(✓) Eye tracker is working. Got {len(samples)} samples in {t}ms.")
        #         self.logger.info(f"Eye tracker is working. Got {len(samples)} samples in {t}ms.")
        #     else:
        #         results.append(f"(✗) Eye tracker is not working. Got 0 samples in {t}ms.")
        #         self.logger.warning(f"Eye tracker is not working. Got 0 samples in {t}ms.")
        # else:
        #     results.append(f"(✗) Eye tracker is not working. Got error: {samples}.")
        #     self.logger.error(f"Eye tracker is not working. Got error: {samples}.")

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
        self.window.units = 'pix'
        # NOTE: If we don't do this early, the eyetracker will override it and we always get a grey background.
        self.window.color = utils.str2tuple(self.stim_params["Display"]["background_color"])

        # Configure a graphics environment (genv) for tracker calibration
        genv = EyeLinkCoreGraphicsPsychoPy(self.tracker.eyelink, self.window)

        # Set background and foreground colors for the calibration target
        foreground_color = utils.str2tuple(self.tracker_params["Calibration"]["target_color"])
        background_color = utils.str2tuple(self.tracker_params["Calibration"]["background_color"])
        genv.setCalibrationColors(foreground_color, background_color)

        # Set up the calibration target
        genv.setTargetType('circle')
        genv.setTargetSize(deg2pix(self.tracker_params["Calibration"]["target_size"], self.monitor))

        # Set up the calibration sounds
        genv.setCalibrationSounds("", "", "")

        return genv

    def gaze_in_square(self, gaze, center, length):
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

    def gaze_in_circle(self, gaze, center, radius):
        """
        Check if the gaze is within a target circle.

        Args:
            gaze (tuple): The gaze coordinates.
            center (tuple): The center of the circle.
            radius (float): The radius of the circle.

        Returns:
            bool: Whether the fixation is within the circle or not.
        """
        offset = utils.get_hypot(gaze, center)
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
        return x - self.window.size[0] / 2, self.window.size[1] / 2 - y

    def cart2mat(self, x, y):
        """
        Convert center coordinates to matrix coordinates.

        Args:
            x (float): The x-coordinate.
            y (float): The y-coordinate.

        Returns:
            tuple: The matrix coordinates.
        """
        return self.window.size[0] / 2 + x, self.window.size[1] / 2 - y

    def establish_fixation(self, method="circle", fix=None):
        """
        Establish fixation.

        Args:
            method (str): The method to use for fixation.
            fix (object): The fixation stimulus.

        Returns:
            bool: Whether the fixation is established or not.
        """
        fix_event = {"fixation_start": pylink.STARTFIX}
        fix_radi = deg2pix(self.exp_params["General"]["valid_fixation_radius"], self.monitor)
        if fix is None:
            fix = self.fix_stim
        start_time = self.trial_clock.getTime()
        while True:
            if self.trial_clock.getTime() - start_time > self.exp_params["General"]["fixation_timeout"]:
                msg = self.tracker.codex_msg("fix", "timeout")
                self.trial_warning(msg)
                return msg

            # Draw fixation
            fix.draw()
            self.window.flip()

            # Check eye events
            events = self.tracker.process_events_online(fix_event)
            if events == self.codex.message("con", "lost"):
                self.trial_error(events)
                return msg

            fixations = []
            for event in events:
                # Only take the last fixation
                gaze_x = event["gaze_start_x"]
                gaze_y = event["gaze_start_y"]
                gx, gy = self.mat2cart(gaze_x, gaze_y)
                if method == "circle":
                    valid = self.gaze_in_circle((gx, gy), fix.pos, fix_radi)
                elif method == "square":
                    valid = self.gaze_in_square((gx, gy), fix.pos, fix_radi)
                else:
                    raise NotImplementedError("Only circle and square methods are implemented.")
                fixations.append(valid)

            if any(fixations):
                msg = self.tracker.codex_msg("fix", "ok")
                return msg

    def reboot_tracker(self):
        """
        """
        self._warn("Rebooting the tracker...")
        self.tracker.reset()
        self.tracker.connect()

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
        fix_radi = deg2pix(self.exp_params["General"]["valid_fixation_radius"], self.monitor)
        fix_events = {"fixation_update": pylink.FIXUPDATE, "fixation_end": pylink.ENDFIX}
        events = self.tracker.process_events_online(fix_events)

        # Check if there was an error
        if events in [self.codex.message("con", "lost"), self.codex.message("con", "term")]:
            self.trial_error(events)
            return events, None

        # Check if the fixation is valid
        if fix_pos is None:
            fix_pos = self.fix_stim.pos
        updates = []
        for event in events:
            if event["event_type"] == "fixation_end":
                msg = self.tracker.codex_msg("fix", "lost")
                self.trial_warning(msg)
                self.tracker.direct_msg(msg, delay=False)
                return msg, events
            else:
                gaze_x = event["gaze_avg_x"]
                gaze_y = event["gaze_avg_y"]
                gx, gy = self.mat2cart(gaze_x, gaze_y)
                if method == "circle":
                    check = self.gaze_in_circle((gx, gy), fix_pos, fix_radi)
                elif method == "square":
                    check = self.gaze_in_square((gx, gy), fix_pos, fix_radi)
                else:
                    raise NotImplementedError("Method is not implemented.")
                updates.append(check)
        if updates and all(updates):
            msg = self.codex.message("fix", "ok")
        else:
            msg = self.codex.message("fix", "bad")
            self.trial_warning(msg)
            self.tracker.direct_msg(msg, delay=False)

        return msg, events

    def monitor_saccade(self, targets_pos, fix_pos=None, fix_method="circle"):
        """
        Look for saccade.

        Args:
            targets_pos (list): The position of the target.
            fix_pos (tuple): The position of the fixation. If None, the position of fixation stimulus is used.
            fix_method (str): The method to use for fixation.

        Returns:
            bool: Whether the fixation is established or not.
        """
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
            return events, None

        # Check if the fixated and if it is valid
        fix_radi = deg2pix(self.exp_params["General"]["valid_fixation_radius"], self.monitor)
        if fix_pos is None:
            fix_pos = self.fix_stim.pos
        updates = []
        landings = []
        sacc_radi = deg2pix(self.exp_params["General"]["valid_saccade_radius"], self.monitor)
        sacc = False
        for event in events:
            if event["event_type"] == "fixation_update":
                gaze_x = event["gaze_avg_x"]
                gaze_y = event["gaze_avg_y"]
                gx, gy = self.mat2cart(gaze_x, gaze_y)
                if fix_method == "circle":
                    check = self.gaze_in_circle((gx, gy), fix_pos, fix_radi)
                elif fix_method == "square":
                    check = self.gaze_in_square((gx, gy), fix_pos, fix_radi)
                else:
                    raise NotImplementedError("Method is not implemented.")
                updates.append(check)
            elif event["event_type"] == "saccade_start":
                sacc = True
            elif event["event_type"] == "saccade_end":
                gaze_x = event["gaze_end_x"]
                gaze_y = event["gaze_end_y"]
                gp = self.mat2cart(gaze_x, gaze_y)
                # check if close to target(s)
                for tp in targets_pos:
                    check = self.gaze_in_circle(gp, tp, sacc_radi)
                    landings.append(check)

        # Check if saccade is valid
        if landings:
            if any(landings):
                msg = self.codex.message("sacc", "good")
                self.trial_info(msg)
                self.tracker.direct_msg(msg, delay=False)
            else:
                msg = self.codex.message("sacc", "bad")
                self.trial_warning(msg)
                self.tracker.direct_msg(msg, delay=False)
        elif sacc:
            msg = self.codex.message("sacc", "onset")
            self.trial_info(msg)
            self.tracker.direct_msg(msg, delay=False)
        else:
            # Check if fixation is valid
            if updates and all(updates):
                msg = self.codex.message("fix", "ok")
                self.trial_info(msg)
            else:
                msg = self.codex.message("fix", "lost")
                self.trial_warning(msg)
                self.tracker.direct_msg(msg, delay=False)

        return msg, events

    def init_events_data(self):
        """
        """
        fname = f"sub-{self.sub_id}_ses-{self.ses_id}_task-{self.task_name}_block-{self.block_id}_EyeEvents.csv"
        events_file = self.ses_data_dir / fname
        if events_file.exists():
            self.block_warning(f"File {events_file} already exists. Renaming the file as backup.")
            backup_file = events_file.with_suffix(".BAK")
            events_file.rename(backup_file)
        self.events_data_file = events_file
        self.events_data = pd.DataFrame(columns=[
            "BlockID", "BlockName", "TrialIndex", "TrialNumber",
            "TrackerLag", "EventType",
            "EventStart_ExpTime_ms", "EventStart_TrackerTime_ms", "EventStart_FrameN",
            "EventEnd_ExpTime_ms", "EventEnd_TrackerTime_ms", "EventEnd_FrameN",
            "EventDuration_ms", "EventDuration_fr", "Event_Period",
            "GazeStartX_px", "GazeStartX_ppd", "GazeStartX_dva",
            "GazeStartY_px", "GazeStartY_ppd", "GazeStartY_dva",
            "GazeEndX_px", "GazeEndX_ppd", "GazeEndX_dva",
            "GazeEndY_px", "GazeEndY_ppd", "GazeEndY_dva",
            "GazeAvgX_px", "GazeAvgX_ppd", "GazeAvgX_dva",
            "GazeAvgY_px", "GazeAvgY_ppd", "GazeAvgY_dva",
            "AmplitudeX_dva", "AmplitudeY_dva",
            "PupilStart_area", "PupilEnd_area", "PupilAvg_area",
            "VelocityStart_dps", "VelocityEnd_dps", "VelocityAvg_dps", "VelocityPeak_dps",
            "Angle_deg", "Angle_rad",
        ])

    def init_samples_data(self):
        """
        """
        fname = f"sub-{self.sub_id}_ses-{self.ses_id}_task-{self.task_name}_block-{self.block_id}_EyeSamples.csv"
        samples_file = self.ses_data_dir / fname
        if samples_file.exists():
            self.block_warning(f"File {samples_file} already exists. Renaming the file as backup.")
            backup_file = samples_file.with_suffix(".BAK")
            samples_file.rename(backup_file)
        self.samples_data_file = samples_file
        self.samples_data = pd.DataFrame(columns=[
            "BlockID", "BlockName", "TrialIndex", "TrialNumber", "TaskPeriod",
            "TrackerLag", "SampleIndex", "SampleEvent",
            "SampleOnset_ExpTime_ms", "SampleOnset_TrackerTime_ms", "SampleOnset_FrameN",
            "GazeX_px", "GazeX_ppd", "GazeX_dva",
            "GazeY_px", "GazeY_ppd", "GazeY_dva",
            "Pupil_area",
        ])

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
        df["BlockName"] = self.all_blocks[int(self.block_id)-1]["name"]
        df["TrialIndex"] = int(self.trial_id - 1)
        df["TrialNumber"] = self.trial_id
        df["TrackerLag"] = self.ms_round(tracker_lag)
        df["EventType"] = data["event_type"]
        # time
        ts = data.get("time_start")
        if ts is not None:
            df["EventStart_TrackerTime_ms"] = self.ms_round(ts)
            df["EventStart_TrialTime_ms"] = self.ms_round(ts - tracker_lag)
            frame_n = self.time_point_to_frame_idx(ts, self.window.frameIntervals)
            df["EventStart_FrameN"] = frame_n
            df["EventStart_Period"] = self.codex.code2msg(self.trial_frames[frame_n])
        te = data.get("time_end")
        if te is not None:
            df["EventEnd_TrackerTime_ms"] = self.ms_round(te)
            df["EventEnd_TrialTime_ms"] = self.ms_round(te - tracker_lag)
            frame_n = self.time_point_to_frame_idx(te, self.window.frameIntervals)
            df["EventEnd_FrameN"] = frame_n
            df["EventEnd_Period"] = self.codex.code2msg(self.trial_frames[frame_n])
        dur = data.get("duration")
        if dur is not None:
            df["EventDuration_ms"] = self.ms_round(dur)
            df["EventDuration_fr"] = self.ms2fr(dur)
        # start gaze
        gsx = data.get("gaze_start_x")
        gsy = data.get("gaze_start_y")
        if gsx is not None and gsy is not None:
            gsx, gsy = self.mat2cart(gsx, gsy)
            df["GazeStartX_px"] = self.val_round(gsx)
            df["GazeStartY_px"] = self.val_round(gsy)
            ppdsx = data.get("ppd_start_x")
            ppdsy = data.get("ppd_start_y")
            if ppdsx is not None and ppdsy is not None:
                df["GazeStartX_ppd"] = self.val_round(ppdsx)
                df["GazeStartY_ppd"] = self.val_round(ppdsy)
                gsx_dva = gsx / ppdsx
                gsy_dva = gsy / ppdsy
                df["GazeStartX_dva"] = self.val_round(gsx_dva)
                df["GazeStartY_dva"] = self.val_round(gsy_dva)
        # end gaze
        gex = data.get("gaze_end_x")
        gey = data.get("gaze_end_y")
        if gex is not None and gey is not None:
            gex, gey = self.mat2cart(gex, gey)
            df["GazeEndX_px"] = self.val_round(gex)
            df["GazeEndY_px"] = self.val_round(gey)
            ppdex = data.get("ppd_end_x")
            ppdey = data.get("ppd_end_y")
            if ppdex is not None and ppdey is not None:
                df["GazeEndX_ppd"] = self.val_round(ppdex)
                df["GazeEndY_ppd"] = self.val_round(ppdey)
                gex = gex / ppdex
                gey = gey / ppdey
                df["GazeEndX_dva"] = self.val_round(gex)
                df["GazeEndY_dva"] = self.val_round(gey)
        # average gaze
        gavgx = data.get("gaze_avg_x")
        gavgy = data.get("gaze_avg_y")
        if gavgx is not None and gavgy is not None:
            gavgx, gavgy = self.mat2cart(gavgx, gavgy)
            df["GazeAvgX_px"] = self.val_round(gavgx)
            df["GazeAvgY_px"] = self.val_round(gavgy)
            ppdavgx = data.get("ppd_avg_x")
            ppdavgy = data.get("ppd_avg_y")
            if ppdavgx is not None and ppdavgy is not None:
                df["GazeAvgX_ppd"] = self.val_round(ppdavgx)
                df["GazeAvgY_ppd"] = self.val_round(ppdavgy)
                gavgx_dva = gavgx / ppdavgx
                gavgy_dva = gavgy / ppdavgy
                df["GazeAvgX_dva"] = self.val_round(gavgx_dva)
                df["GazeAvgY_dva"] = self.val_round(gavgy_dva)
        # amplitude
        ampx = data.get("amp_x")
        ampy = data.get("amp_y")
        if ampx is not None and ampy is not None:
            df["AmplitudeX_dva"] = self.val_round(ampx)
            df["AmplitudeY_dva"] = self.val_round(ampy)
        # angle
        ang = data.get("angle")
        if ang is not None:
            df["Angle_deg"] = self.val_round(ang)
            df["Angle_rad"] = self.val_round(np.radians(ang))
        # velocity
        sv = data.get("velocity_start")
        if sv is not None:
            df["VelocityStart_dps"] = self.val_round(sv)
        ev = data.get("velocity_end")
        if ev is not None:
            df["VelocityEnd_dps"] = self.val_round(ev)
        avgv = data.get("velocity_avg")
        if avgv is not None:
            df["VelocityAvg_dps"] = self.val_round(avgv)
        pv = data.get("velocity_peak")
        if pv is not None:
            df["VelocityPeak_dps"] = self.val_round(pv)
        # pupil
        ps = data.get("pupil_start")
        if ps is not None:
            df["PupilStart_area"] = self.val_round(ps)
        pe = data.get("pupil_end")
        if pe is not None:
            df["PupilEnd_area"] = self.val_round(pe)
        avgp = data.get("pupil_avg")
        if avgp is not None:
            df["PupilAvg_area"] = self.val_round(avgp)

        # Concatenate the data
        self.events_data = pd.concat([self.events_data, df], ignore_index=True)

    def init_edf_files(self):
        """
        """
        # Display file
        fname = f"sub-{self.sub_id}_ses-{self.ses_id}_task-{self.task_name}_block-{self.block_id}.edf"
        edf_display_file = self.ses_data_dir / fname
        if edf_display_file.exists():
            self.block_warning(f"File {edf_display_file} already exists. Renaming the file as backup.")
            edf_display_file.rename(edf_display_file.with_suffix(".BAK"))
        self.edf_display_file = edf_display_file

        # Host file
        self.edf_host_file = f"{self.sub_id}_{self.ses_id}_{self.block_id}.edf"
        status = self.tracker.open_edf_file(self.edf_host_file)
        if status == self.codex.message("edf", "init"):
            self.block_info(status)
        else:
            self.block_error(status)

    def init_block_data(self):
        """
        """
        super().init_block_data()
        self.init_events_data()
        self.init_samples_data()
        self.init_edf_files()

    def save_events_data(self):
        """
        """
        self.events_data.to_csv(self.events_data_file, sep=',', index=False)

    def save_samples_data(self):
        """
        """
        self.samples_data.to_csv(self.samples_data_file, sep=',', index=False)

    def goodbye(self, raise_error=None):
        """
        End the experiment. Close the window and the tracker.
        If raising an error (quitting in the middle of the experiment), save as much data as possible.

        Args:
            raise_error (str): The error message to raise.
        """
        # Close the window
        self.window.close()

        # Quit
        if raise_error is not None:
            # Close the eye tracker
            self.tracker.codex_msg("exp", "term")
            self.tracker.eyelink.terminalBreak(1)
            self.tracker.terminate(self.edf_host_file, self.edf_display_file)
            # Log the error
            self.logger.critical(raise_error)
            # Save as much as you can
            try:
                self.save_stim_data()
                self.save_behav_data()
                self.save_frame_data()
                self.save_events_data()
                self.save_samples_data()
                self.save_log_data()
            except AttributeError as e:
                self.logger.error(f"Error saving data: {e}")
                # Raise the error
                raise SystemExit(f"Experiment ended with error: {raise_error}")
        else:
            # Close the eye tracker
            status = self.tracker.terminate(self.edf_host_file, self.edf_display_file)
            if status == self.codex.message("file", "ok"):
                self.logger.info("Eye tracker file closed.")
            else:
                # Log
                self.logger.error(status)
            self.logger.info("Bye Bye Experiment.")
            core.quit()
