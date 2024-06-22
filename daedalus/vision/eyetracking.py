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
import numpy as np
import pandas as pd

from daedalus.psyphy import PsychoPhysicsExperiment, PsychoPhysicsDatabase
from daedalus.data.database import EyeTrackingEvent, EyeTrackingSample
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
        self.edf_display_file = None
        self.edf_host_file = None

        # Monitoring parameters
        self.fix_radi = deg2pix(self.exp_params["General"]["fix_radi"])
        fixation_check_frequency = self.exp_params["General"]["fix_check_interval"]
        self.fix_check_interval = self.ms2fr(1000 / fixation_check_frequency)

    def init_tracker(self):
        """
        Initialize the eye tracker.
        """
        # Connect to the eye tracker
        self.tracker = MyeLink(self.name, self.tracker_params, self.debug)
        self.tracker.connect()
        self.tracker.configure(display_name=self.platform)

        # Set it to idle mode
        self.tracker.go_offline()

        # Open graphics environment
        genv = self.make_graphics_env()
        w, h = self.window.size
        self.tracker.set_calibration_graphics(w, h, genv)

        # Log the initialization
        self.logger.info("Eyetracker is locked and loaded and ready to go.")
        self.tracker.send_cmd("record_status_message 'Eyetracker configured'")
        self.tracker.codex_msg("tracker", "init")

    def prepare_block(self, block_id, repeat=False, calib=False):
        """
        Prepares the block for the experiment.
        """
        # Set the ID of the block
        self.block_id = self._fix_id(block_id)

        # Make sure tracker is not breaking
        self.tracker.eyelink.terminalBreak(0)

        # Take the tracker offline
        self.tracker.go_offline()

        # Show block info as text
        n_blocks = len(self.get_reamining_blocks())
        if repeat:
            msg = f"You are repeating block {self.block_id}/{n_blocks}."
            calib = True
        else:
            msg = f"You are about to begin block {self.block_id}/{n_blocks}."

        # Calibrate if needed
        if calib:
            calib_res = self.run_calibration(msg)
            if calib_res == self.codex.message("calib", "term"):
                self.show_msg("Skipping calibration.", msg_type="warning")
            elif calib_res == self.codex.message("calib", "fail"):
                self.show_msg("Calibration failed. Please try again.", msg_type="warning")
                self.prepare_block(self.block_id, repeat, calib)
            else:
                msg_ = ["Calibration is done."]
                msg_.append(msg)
                msg_.append("Press Space to start the block.")
                self.show_msg("\n\n".join(msg_))
        else:
            msg_ = [msg, "Press Space to start the block."]
            self.show_msg("\n\n".join(msg_))

        # Reset the block clock
        self.block_clock.reset()
        # Initialize the block data
        self.init_block_data()
        # Log the start of the block
        self.log_info(self.codex.message("block", "init"))
        self.log_info(f"BLOCKID_{self.block_id}")
        self.tracker.direct_msg(f"BLOCKID {self.block_id}")
        self.tracker.send_cmd(f"record_status_message 'Block {self.block_id}/{n_blocks}.'")

    def prepare_trial(self, trial_id, fixation):
        """
        """
        self.trial_id = self._fix_id(trial_id)

        # Establish fixation
        fix_status = self.establish_fixation(fixation)
        if fix_status:
            # Drift correction
            fix_pos = self.cart2mat(*fixation.pos)
            msg = self.tracker.drift_correct(fix_pos)
            if msg == self.codex.message("con", "lost"):
                self.handle_connection_loss()
            else:
                if msg == self.codex.message("drift", "ok"):
                    self.log_info(msg)
                    recalib = False
                else:
                    self.log_warning(msg)
                    recalib = True
                # Start recording
                on_status = self.tracker.come_online()
                if on_status == self.codex.message("rec", "init"):
                    self.log_info(on_status)
                    self.trial_clock.reset()
                    self.log_info(f"TRIALID_{self.trial_id}")
                    self.tracker.direct_msg(f"TRIALID {trial_id}")
                    self.tracker.send_cmd(f"record_status_message 'Block {self.block_id}, Trial {self.trial_id}.'")
                    self.tracker.direct_msg("!V CLEAR 128 128 128")
                else:
                    self.log_error(on_status)
        # Fixation timeout
        else:
            recalib = True

        return recalib

    def handle_not_recording(self, error):
        """
        """
        msg = f"Eyetracker is not recording.\n{error}\nReconnect? (Space)"
        resp = self.show_msg(msg, msg_type="warning")
        if resp == "space":
            self.init_tracker()

    def handle_connection_loss(self):
        """
        """
        msg = "Connection to the eye tracker is lost. Reconnect? (Space)"
        resp = self.show_msg(msg, msg_type="warning")
        if resp == "space":
            self.init_tracker()

    def wrap_trial(self, repeat):
        """
        """
        super().wrap_trial()

        # Messages
        if repeat:
            self.tracker.direct_msg(f"TRIAL_RESULT {pylink.REPEAT_TRIAL}")
        else:
            self.tracker.direct_msg(f"TRIAL_RESULT {pylink.TRIAL_OK}")
        self.tracker.direct_msg("!V CLEAR 128 128 128")

        # Stop recording
        self.tracker.go_offline()
        self.tracker.reset()

    def wrap_block(self):
        """
        """
        super.wrap_block()

        # Stop recording
        self.tracker.codex_msg("block", "fin")
        self.tracker.go_offline()
        self.tracker.reset()

        # Save the data
        self.save_events_data()
        self.save_samples_data()

    def stop_block(self):
        """
        Stop the block.
        """
        # Stop recording
        self.tracker.direct_msg(f"TRIAL_RESULT {pylink.TRIAL_ERROR}")
        self.tracker.codex_msg("block", "stop")
        self.tracker.eyelink.terminalBreak(1)
        self.tracker.go_offline()
        self.tracker.reset()
        super().stop_block()

    def run_calibration(self, msg=None):
        """
        Run the calibration for the eyetracker.

        Args:
            msg (str): The message to show before calibration.
        """
        # Message
        if msg is None:
            txt = []
        else:
            txt = [msg]
        txt.append("In the next screen, press C to calibrate the eyetracker.")
        txt.append("Once the calibration is done, press Enter to accept the calibration and resume the experiment.")
        txt.append("Press Space to continue.")
        resp = self.show_msg("\n\n".join(txt))

        # Run calibration
        self.tracker.send_cmd("record_status_message 'In calibration'")
        if resp == "escape":
            status = self.tracker.codex_msg("calib", "term")
            self.log_warning(status)
        elif resp == "space":
            self.log_info(self.codex.message("calib", "init"))
            # Take the tracker offline
            self.tracker.go_offline()
            # Start calibration
            res = self.tracker.calibrate()
            if res == self.codex.message("calib", "ok"):
                self.log_info(res)
                self.msg_stim.text = "Calibration is done. Press Enter to accept the calibration and resume the experiment."
                while self.tracker.eyelink.inSetup():
                    self.msg_stim.draw()
                    self.window.flip()
                status = res
            else:
                self.log_error(res)
                self.show_msg(f"Calibration error:\n\n{status}", msg_type="warning")
                status = self.codex.message("calib", "fail")

        return status

    def tracker_check(self):
        """
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
            self.log_error(on_status)

        # Check the eye
        eye_status = self.tracker.check_eye()
        if eye_status == self.codex.message("eye", "good"):
            results.append("(✓) Eye checks out.")
            self.logger.info("Eye checks out.")
        else:
            results.append("(✗) Eye mismatch.")
            self.logger.error("Eye mismatch.")

        # Check the samples
        t = 500
        self.tracker.delay()
        samples = self.tracker.process_samples_online()
        if isinstance(samples, list):
            if len(samples) > 0:
                results.append(f"(✓) Eye tracker is working. Got {len(samples)} samples in {t}ms.")
                self.logger.info(f"Eye tracker is working. Got {len(samples)} samples in {t}ms.")
            else:
                results.append(f"(✗) Eye tracker is not working. Got 0 samples in {t}ms.")
                self.logger.warning(f"Eye tracker is not working. Got 0 samples in {t}ms.")
        else:
            results.append(f"(✗) Eye tracker is not working. Got error: {samples}.")
            self.logger.error(f"Eye tracker is not working. Got error: {samples}.")

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

        # Set up the calibration sounds
        genv.setCalibrationSounds("", "", "")
        genv.setDriftCorrectSounds("", "", "")

        return genv

    def set_graphics_env(self):
        """
        Make and set the graphics environment for the calibration.
        """
        # Screen size
        width, height = self.window.size
        # Make the graphics environment
        genv = self.make_graphics_env()
        # Set the calibration graphics
        self.tracker.set_calibration_graphics(width, height, genv)

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

    def establish_fixation(self, fixation, method="circle"):
        """
        Establish fixation.

        Args:
            fixation (object): The fixation stimulus.
            timeout (float): The time to wait for fixation.
            radi (float): The radius of the circle.

        Returns:
            bool: Whether the fixation is established or not.
        """
        fix_event = {"fixation_start": pylink.STARTFIX}
        fx, fy = fixation.pos

        start_time = self.trial_clock.getTime()
        while True:
            if self.trial_clock.getTime() - start_time > self.exp_params["General"]["fixation_timeout"]:
                msg = self.tracker.codex_msg("fix", "timeout")
                self.log_error(msg)
                return False

            # Draw fixation
            fixation.draw()
            self.window.flip()

            # Check eye events
            eye_events = self.tracker.process_events_online(fix_event)
            if isinstance(eye_events, int):
                self.log_error(eye_events)
                return False
            elif eye_events["fixation_start"]:
                # Only take the last fixation
                gaze_x = eye_events["fixation_start"][-1]["gaze_start_x"]
                gaze_y = eye_events["fixation_start"][-1]["gaze_start_y"]
                gx, gy = self.mat2cart(gaze_x, gaze_y)
                if method == "circle":
                    valid = self.check_fixation_in_circle(fx, fy, gx, gy, self.fix_radi)
                elif method == "square":
                    valid = self.check_fixation_in_square(fixation.pos, self.fix_radi, (gx, gy))
                else:
                    raise NotImplementedError("Only circle and square methods are implemented.")
                if valid:
                    msg = self.tracker.codex_msg("fix", "ok")
                    self.log_info(msg)
                    return True

    def reboot_tracker(self):
        """
        """
        self._warn("Rebooting the tracker...")
        self.tracker.reset()
        self.tracker.connect()

    def monitor_fixation(self, fix_pos=(0, 0), method="circle"):
        """
        Monitor fixation.

        Args:
            fixation (object): The fixation stimulus.
            timeout (float): The time to wait for fixation.
            radi (float): The radius of the circle.

        Returns:
            bool: Whether the fixation is established or not.
        """
        fix_events = {"fixation_update": pylink.FIXUPDATE, "fixation_end": pylink.ENDFIX}
        events = self.tracker.process_events_online(fix_events)
        fx, fy = fix_pos

        # Check if there was an error
        if isinstance(events, int):
            self.log_error(events)
            return events, events

        # Check if the fixation is lost
        if events["fixation_end"]:
            msg = self.tracker.codex_msg("fix", "lost")
            self.log_warn(msg)
            return msg, events

        # Check if the fixation is valid
        valid_updates = []
        if events["fixation_update"]:
            for event in events["fixation_update"]:
                gaze_x = event["gaze_x"]
                gaze_y = event["gaze_y"]
                gx, gy = self.mat2cart(gaze_x, gaze_y)
                if method == "circle":
                    check = self.check_fixation_in_circle(fx, fy, gx, gy, self.fix_radi)
                elif method == "square":
                    check = self.check_fixation_in_square(fix_pos, self.fix_radi, (gx, gy))
                else:
                    raise NotImplementedError("Method is not implemented.")
                valid_updates.append(check)
            if all(valid_updates):
                msg = self.tracker.codex_msg("fix", "ok")
                self.log_info(msg)
            else:
                msg = self.tracker.codex_msg("fix", "lost")
                self.log_warn(msg)
        else:
            msg = self.tracker.codex_msg("con", "lost")
            self.log_error(msg)

        return msg, events

    def look_for_saccade(self, fix_pos=(0, 0), method="circle"):
        """
        Look for saccade.

        Args:
            fixation (object): The fixation stimulus.
            timeout (float): The time to wait for fixation.
            radi (float): The radius of the circle.

        Returns:
            bool: Whether the fixation is established or not.
        """
        fix_events = {"fixation_update": pylink.FIXUPDATE, "fixation_end": pylink.ENDFIX}
        events = self.tracker.process_events_online(fix_events)
        fx, fy = fix_pos

        # Check if there was an error
        if isinstance(events, int):
            self.log_error(events)
            return events, events

        # Check if the fixation is lost
        if events["fixation_end"]:
            msg = self.tracker.codex_msg("fix", "lost")
            self.log_warn(msg)
            return msg, events

        # Check if the fixation is valid
        valid_updates = []
        if events["fixation_update"]:
            for event in events["fixation_update"]:
                gaze_x = event["gaze_x"]
                gaze_y = event["gaze_y"]
                gx, gy = self.mat2cart(gaze_x, gaze_y)
                if method == "circle":
                    check = self.check_fixation_in_circle(fx, fy, gx, gy, self.fix_radi)
                elif method == "square":
                    check = self.check_fixation_in_square(fix_pos, self.fix_radi, (gx, gy))
                else:
                    raise NotImplementedError("Method is not implemented.")
                valid_updates.append(check)
            if all(valid_updates):
                msg = self.log_info(self.codex.message("fix", "ok"))
            else:
                msg = self.log_warn(self.codex.message("fix", "lost"))
        else:
            msg = self.log_warn(self.codex.message("con", "lost"))

        return msg, events

    def init_events_data(self):
        """
        """
        columns = [
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
        ]
        self.events_data = pd.DataFrame(columns=columns)

    def _init_events_file(self, file_path):
        """
        """
        events_file = self.ses_data_dir / f"sub-{self.subj_id}_ses-{self.ses_id}_task-{self.task_name}_block-{self.block_id}_EyeEvents.csv"
        if events_file.exists():
            self.log_warn(f"File {events_file} already exists. Renaming the file as backup.")
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
        samples_file = self.ses_data_dir / f"sub-{self.subj_id}_ses-{self.ses_id}_task-{self.task_name}_block-{self.block_id}_EyeSamples.csv"
        if samples_file.exists():
            self.log_warn(f"File {samples_file} already exists. Renaming the file as backup.")
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

    def add_to_events_data(self, data, event_type, tracker_lag):
        """
        """
        df = pd.DataFrame()
        df["BlockID"] = self.block_id
        df["BlockName"] = self.task.all_blocks[self.block_id-1]["name"]
        df["TrialIndex"] = self.trial_id - 1
        df["TrialNumber"] = self.trial_id
        df["TrackerLag"] = self.ms_round(tracker_lag)
        df["EventType"] = event_type
        # time
        ts = data.get("time_start")
        if ts is not None:
            df["EventStart_TrackerTime_ms"] = self.ms_round(ts)
            df["EventStart_TrialTime_ms"] = self.ms_round(ts - tracker_lag)
            frame_n = self.time_point_to_frame_idx(ts, self.window.frameIntervals)
            df["EventStart_FrameN"] = frame_n
            df["EventStart_Period"] = self.task.get_period_name(self.task.trial_frames[frame_n])
        te = data.get("time_end")
        if te is not None:
            df["EventEnd_TrackerTime_ms"] = self.ms_round(te)
            df["EventEnd_TrialTime_ms"] = self.ms_round(te - tracker_lag)
            frame_n = self.time_point_to_frame_idx(te, self.window.frameIntervals)
            df["EventEnd_FrameN"] = frame_n
            df["EventEnd_Period"] = self.task.get_period_name(self.task.trial_frames[frame_n])
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
        edf_display_file = self.ses_data_dir / f"sub-{self.subj_id}_ses-{self.ses_id}_task-{self.task_name}_block-{self.block_id}.edf"
        if edf_display_file.exists():
            self.log_warn(f"File {edf_display_file} already exists. Renaming the file as backup.")
            edf_display_file.rename(edf_display_file.with_suffix(".BAK"))
        self.edf_display_file = edf_display_file

        # Host file
        self.edf_host_file = f"{self.subj_id}_{self.ses_id}_{self.block_id}.edf"
        status = self.tracker.open_edf_file(self.edf_host_file)
        if status == self.codex.message("edf", "init"):
            self.log_info(status)
        else:
            self.log_error(status)

    def init_block_data(self):
        """
        """
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
            self.tracker.terminate()
            # Log the error
            self.log_critical(raise_error)
            # Save as much as you can
            self.save_stim_data()
            self.save_behav_data()
            self.save_frame_data()
            self.save_events_data()
            self.save_samples_data()
            self.save_log_data()
            # Raise the error
            raise SystemExit(f"Experiment ended with error: {raise_error}")
        else:
            # Close the eye tracker
            status = self.tracker.terminate()
            if status == self.codex.message("file", "ok"):
                self.log_info("Eye tracker file closed.")
            else:
                # Log
                self.log_error(status)
            self.log_info("Bye Bye Experiment.")
            core.quit()


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
