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
import pandas as pd

from daedalus.codes import Codex


class DataManager:
    """
    DataManager class to handle data operations for daedalus

    Args:
        name (str): Name of the module
        root (str): Root directory
        version (str): Version
    """
    def __init__(self, name, params, version):

        # Setup
        self.name = name
        self.params = params
        self.version = version
        self.codex = Codex()

        # IDs
        self._sub_id = None
        self._ses_id = None
        self._task_id = None

        # Data
        self.participants = None
        self.database = None
        self.stimuli = None
        self.behavior = None
        self.frames = None
        self.eye_events = None
        self.eye_samples = None

    @property
    def sub_id(self):
        return self._sub_id

    @sub_id.setter
    def sub_id(self, value):
        self._sub_id = int(value)

    @property
    def ses_id(self):
        return self._ses_id

    @ses_id.setter
    def ses_id(self, value):
        self._ses_id = int(value)

    @property
    def task_id(self):
        return self._task_id

    @task_id.setter
    def task_id(self, value):
        self._task_id = value

    def init_participants(self):
        """
        Initialize the participants for the experiment.
        """
        columns = ["Session", "Task", "Completed", "Experimenter", "Experiment", "Version"]
        columns += [k.replace("_", "") for k in self.params["Subjects"].keys()]
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
        if self._sub_id not in pids:
            self.participants = pd.concat([self.participants, sub_df], ignore_index=False)
        else:
            duplicate = self.participants.loc[
                (self.participants["PID"] == self._sub_id) &
                (self.participants["Session"] == self._ses_id) &
                (self.participants["Task"] == self._task_id)
            ]
            if duplicate.shape[0] == 0:
                self.participants = pd.concat([self.participants, sub_df], ignore_index=False)
            else:
                return self.codex.message("sub", "dup")

    def update_participant(self, col, value):
        """
        Updates the participant information.

        Args:
            sub_info (dict): The subject information.
        """
        self.participants.loc[
            (self.participants["PID"] == int(self._sub_id)) &
            (self.participants["Session"] == int(self._ses_id)) &
            (self.participants["Task"] == self._task_id), col] = value

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
            "TrialIndex", "TrialNumber", "TrialDuration_ms", "TrialDuration_frames", "TrialRepeated"
        ])

    def init_stimuli(self):
        """
        Initialize the stimuli for the experiment.
        """
        self.stimuli = pd.DataFrame(columns=[
            "SubjectID", "SessionID", "TaskID", "BlockID", "BlockName", "BlockDuration_sec",
            "TrialIndex", "TrialNumber", "TrialDuration_ms", "TrialDuration_frames", "TrialRepeated"
        ])

    def init_frames(self):
        """
        Initialize the frames for the experiment.
        """
        self.frames = pd.DataFrame(columns=[
            "TrialIndex",
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

    def add_block_info(self, block):
        """
        Add block information to the data.

        Args:
            block (Block): The block object.
        """
        self.behavior["BlockID"] = block.id
        self.behavior["BlockName"] = block.name
        self.behavior["BlockDuration_sec"] = block.duration

        self.stimuli["BlockID"] = block.id
        self.stimuli["BlockName"] = block.name
        self.stimuli["BlockDuration_sec"] = block.duration

        self.frames["BlockID"] = block.id
        self.frames["BlockName"] = block.name

        self.eye_events["BlockID"] = block.id
        self.eye_events["BlockName"] = block.name

        self.eye_samples["BlockID"] = block.id
        self.eye_samples["BlockName"] = block.name

    def reset(self):
        """
        Reset the data manager.
        """
        self.behavior = None
        self.stimuli = None
        self.frames = None
        self.eye_events = None
        self.eye_samples = None

    def not_empty(self, df):
        """
        Check if the dataframe is empty.

        Args:
            df (dataframe): The dataframe to check.
        """
        if (not df.empty) and (not df.isna().all().all()):
            return True
        return False

    def collect_trial(self, df, data_type):
        """
        Collect the trial data.

        Args:
            df (dataframe): The trial data to collect.
            data_type (str): The type of data to collect.
        """
        if self.not_empty(df):
            if data_type == "behavior":
                if not self.behavior.empty and not self.behavior.isna().all().all():
                    self.behavior = pd.concat([self.behavior, df], ignore_index=True)
                else:
                    self.behavior = df
            elif data_type == "stimuli":
                if not self.stimuli.empty and not self.stimuli.isna().all().all():
                    self.stimuli = pd.concat([self.stimuli, df], ignore_index=True)
                else:
                    self.stimuli = df
            elif data_type == "frames":
                if not self.frames.empty and not self.frames.isna().all().all():
                    self.frames = pd.concat([self.frames, df], ignore_index=True)
                else:
                    self.frames = df
            elif data_type == "eye_events":
                if not self.eye_events.empty and not self.eye_events.isna().all().all():
                    self.eye_events = pd.concat([self.eye_events, df], ignore_index=True)
                else:
                    self.eye_events = df
            elif data_type == "eye_samples":
                if not self.eye_samples.empty and not self.eye_samples.isna().all().all():
                    self.eye_samples = pd.concat([self.eye_samples, df], ignore_index=True)
                else:
                    self.eye_samples = df
            else:
                raise ValueError(f"Data type {data_type} not recognized.")
        else:
            return self.codex.message("data", "null")

    def save_stimuli(self, file_path):
        """
        Save the stimulus data.
        """
        self.stimuli.to_csv(file_path, sep=",", index=False)

    def save_behavior(self, file_path):
        """
        Save the behavioral data.
        """
        self.behavior.to_csv(file_path, sep=",", index=False)

    def save_frames(self, file_path):
        """
        Save the frame data.
        """
        self.frames.to_csv(file_path, sep=",", index=False)

    def save_eye_events(self, file_path):
        """
        Save the eye events.
        """
        self.eye_events.to_csv(file_path, sep=",", index=False)

    def save_eye_samples(self, file_path):
        """
        Save the eye samples.
        """
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
