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

    def add_session(self):
        pass

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
        pass

    def init_frames(self):
        """
        Initialize the frames for the experiment.
        """
        pass

    def init_eye_events(self):
        """
        Initialize the eye events for the experiment.
        """
        pass

    def init_eye_samples(self):
        """
        Initialize the eye samples for the experiment.
        """
        pass

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