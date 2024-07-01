#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =================================================================================================== #
#
#
#                    SCRIPT: data.py
#
#
#               DESCRIPTION: Data manager
#
#
#                      RULE: DAYW
#
#
#
#                   CREATOR: Sharif Saleki
#                      TIME: 07-01-2024-7810598105114117
#                     SPACE: Dartmouth College, Hanover, NH
#
# =================================================================================================== #
import numpy as np
import pandas as pd

from daedalus.codes import Codex
from daedalus.utils import time_index_from_sum


class DataManager:
    """
    DataManager class to handle data operations for daedalus

    Args:
        name (str): Name of the module
        root (str): Root directory
        version (str): Version
    """
    def __init__(self, name, root, version):

        # Setup
        self.name = name
        self.root = root
        self.version = version
        self.codex = Codex()

        # IDs
        self.sub_id = None
        self.ses_id = None
        self.task_id = None

        # Data
        self.participants = None
        self.database = None
        self.stimuli = None
        self.behavior = None
        self.frames = None
        self.eye_events = None
        self.eye_samples = None

    def init_participants(self):
        """
        Initialize the participants for the experiment.
        """
        columns = ["Session", "Task", "Completed", "Experimenter", "Experiment", "Version"]
        columns += [k.replace("_", "") for k in self.settings.exp["Subjects"].keys()]
        self.participants = pd.DataFrame(columns=columns)

    def load_participants(self, file_path):
        """
        Load the participants file.

        Args:
            file_path (str): The path to the participants file.
        """
        if file_path.exists():
            self.participants = pd.read_csv(file_path, sep="\t")
        else:
            self.init_participants()

    def add_participant(self, sub_info):
        """
        Saves the subject info to the participants file.

        Args:
            subject_info (dict or DataFrame): The subject info as a dictionary.
        """
        if isinstance(sub_info, dict):
            info = {k: [v] for k, v in sub_info.items()}
            sub_df = pd.DataFrame.from_dict(info, orient="columns")
        else:
            sub_df = sub_info

        pids = self.participants["PID"].values.astype(int)
        sub_id = int(sub_df["PID"].values[0])
        ses_id = int(sub_df["Session"].values[0])
        task_id = sub_df["Task"].values[0]
        if sub_id not in pids:
            self.participants = pd.concat([self.participants, sub_df], ignore_index=False)
        else:
            duplicate = self.participants.loc[
                (self.participants["PID"] == sub_id) &
                (self.participants["Session"] == ses_id) &
                (self.participants["Task"] == task_id)
            ]
            if duplicate.shape[0] == 0:
                self.participants = pd.concat([self.participants, sub_df], ignore_index=False)
            else:
                return self.codex.message("sub", "dup")
        self.sub_id = sub_id
        self.ses_id = ses_id
        self.task_id = task_id

    def update_participant(self, col, value):
        """
        Updates the participant information.

        Args:
            sub_info (dict): The subject information.
        """
        self.participants.loc[
            (self.participants["PID"] == self.sub_id) &
            (self.participants["Session"] == self.ses_id) &
            (self.participants["Task"] == self.task_id), col] = value

    def save_participants(self, file_path):
        """
        Save the participants file.

        Args:
            file_path (str): The path to the participants file.
        """
        self.participants.to_csv(file_path, sep="\t", index=False)

    def init_behavior(self):
        """
        Initialize the behavioral data for the experiment.
        """
        self.behavior = pd.DataFrame(columns=[
            "SubjectID", "SessionID", "TaskID", "BlockID", "BlockName", "BlockDuration_sec",
            "TrialIndex", "TrialNumber", "TrialDuration_ms", "TrialDuration_frames",
        ])

    def init_stimuli(self):
        """
        Initialize the stimuli for the experiment.
        """
        self.stimuli = pd.DataFrame(columns=[
            "SubjectID", "SessionID", "TaskID", "BlockID", "BlockName", "BlockDuration_sec",
            "TrialIndex", "TrialNumber", "TrialDuration_ms", "TrialDuration_frames",
        ])

    def init_frames(self):
        """
        Initialize the frames for the experiment.
        """
        self.frames = pd.DataFrame(columns=[
            "SubjectID", "SessionID", "TaskID","BlockID", "TrialIndex",
            "Period",
            "FrameIndex", "FrameDuration_ms", "Frame_TrialTime_ms"
        ])

    def init_eye_events(self):
        """
        Initialize the eye events for the experiment.
        """
        self.eye_events = pd.DataFrame(columns=[
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

    def init_eye_samples(self):
        """
        Initialize the eye samples for the experiment.
        """
        self.eye_samples = pd.DataFrame(columns=[
            "BlockID", "BlockName", "TrialIndex", "TrialNumber", "TaskPeriod",
            "TrackerLag", "SampleIndex", "SampleEvent",
            "SampleOnset_ExpTime_ms", "SampleOnset_TrackerTime_ms", "SampleOnset_FrameN",
            "GazeX_px", "GazeX_ppd", "GazeX_dva",
            "GazeY_px", "GazeY_ppd", "GazeY_dva",
            "Pupil_area",
        ])

    def add_eye_events(self, df):
        """
        Add a dictionary of data to the events dataframe.

        Args:
            df (dataframe): The data to add to the events dataframe.
        """
        self.eye_events = pd.concat([self.eye_events, df], ignore_index=True)

    def save_stimuli(self, file_path, block_id=None):
        """
        Save the stimulus data.
        """
        if block_id is not None:
            self.stimuli.loc[self.stimuli["BlockID"] == int(block_id)].to_csv(file_path, sep=",", index=False)
        else:
            self.stimuli.to_csv(file_path, sep=",", index=False)

    def save_behavior(self, file_path, block_id=None):
        """
        Save the behavioral data.
        """
        if block_id is not None:
            self.behavior.loc[self.behavior["BlockID"] == int(block_id)].to_csv(file_path, sep=",", index=False)
        else:
            self.behavior.to_csv(file_path, sep=",", index=False)

    def save_frames(self, file_path, block_id=None):
        """
        Save the frame data.
        """
        if block_id is not None:
            self.frames.loc[self.frames["BlockID"] == int(block_id)].to_csv(file_path, sep=",", index=False)
        else:
            self.frames.to_csv(file_path, sep=",", index=False)

    def save_eye_events(self, file_path, block_id=None):
        """
        Save the eye events.
        """
        if block_id is not None:
            self.eye_events.loc[self.eye_events["BlockID"] == int(block_id)].to_csv(file_path, sep=",", index=False)
        else:
            self.eye_events.to_csv(file_path, sep=",", index=False)

    def save_eye_samples(self, file_path, block_id=None):
        """
        Save the eye samples.
        """
        if block_id is not None:
            self.eye_samples.loc[self.eye_samples["BlockID"] == int(block_id)].to_csv(file_path, sep=",", index=False)
        else:
            self.eye_samples.to_csv(file_path, sep=",", index=False)

    def load_ses_behavior(self, beh_files):
        """
        Load the behavioral data.
        """
        self.behavior = pd.concat([pd.read_csv(file) for file in beh_files], ignore_index=True)

    def load_ses_stimuli(self, stim_files):
        """
        Load the stimulus data.
        """
        self.stimuli = pd.concat([pd.read_csv(file) for file in stim_files], ignore_index=True)

    def load_ses_frames(self, frame_files):
        """
        Load the frame data.
        """
        self.frames = pd.concat([pd.read_csv(file) for file in frame_files], ignore_index=True)

    def load_ses_eye_events(self, eye_event_files):
        """
        Load the eye events.
        """
        self.eye_events = pd.concat([pd.read_csv(file) for file in eye_event_files], ignore_index=True)

    def load_ses_eye_samples(self, eye_sample_files):
        """
        Load the eye samples.
        """
        self.eye_samples = pd.concat([pd.read_csv(file) for file in eye_sample_files], ignore_index=True)

    # def init_database(self):
    #     """
    #     Initialize the database for the experiment.
    #     """
    #     self.db = PsychoPhysicsDatabase(self.files["database"], self.settings["Study"]["ID"])
    #     self.db.add_experiment(
    #         title=self.settings["Study"]["Title"],
    #         shorthand=self.settings["Study"]["Shorthand"],
    #         repository=self.settings["Study"]["Repository"],
    #         version=self.version,
    #         description=self.settings["Study"]["Description"],
    #         n_subjects=self.exp_params["NumSubjects"],
    #         n_sessions=self.exp_params["NumSessions"]
    #     )