#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =================================================================================================== #
#
#
#                    SCRIPT: files.py
#
#
#               DESCRIPTION: Filemanager
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
from pathlib import Path


class FileManager:
    """
    FileManager class to handle file operations for the vision module

    Args:
        root (str): Root directory for the vision module

    Attributes:
        root (str): Root directory for the vision module
        data_dir (str): Data directory for the vision module
    """
    def __init__(self, name, root, version):

        # Setup
        self.name = name
        self.root = Path(root)
        self.version = version

        # Directories
        self.dirs = DirectoryManager(root)

        # Config file
        self.participants = self.dirs.data / "participants.tsv"
        self.database = self.dirs.data / f'{self.name}_v{self.version}.db'
        if not self.database.exists():
            self.database.touch()

        self.log = None
        self.stim_data = None
        self.behavior = None
        self.frames = None
        self.eye_events = None
        self.eye_samples = None

    def add_session(self, sub_id, ses_id):

        # Add subject
        self.dirs.add_subject(sub_id)

        # Add session
        self.dirs.add_session(ses_id)

        # Log file
        self.log = self.dirs.logs / f"sub-{sub_id}_ses-{ses_id}_v{self.version}.log"
        if self.log.exists():
            self.log.unlink()
        self.log.touch()

    def add_block(self, sub_id, ses_id, task_id, block_id):

        fname = f"sub-{sub_id}_ses-{ses_id}_task-{task_id}_block-{block_id}_behavioral.csv"
        behav_file = self.dirs.ses / fname
        if behav_file.exists():
            behav_file.rename(behav_file.with_suffix(".BAK"))
            return self.codex.message("data", "dup")


class DirectoryManager:
    """
    DirectoryManager class to handle directory operations for the vision module

    Args:
        root (str): Root directory for the vision module

    Attributes:
        root (str): Root directory for the vision module
        data_dir (str): Data directory for the vision module
    """
    def __init__(self, root, version):

        # Setup
        self.root = Path(root)
        self.version = version

        self.config = root / "config"

        self.stimuli = root / "stimuli"

        self.fonts = root.parents[2] / "tools" / "fonts"

        self.data = root / "data" / f"v{version}"
        self.data.mkdir(parents=True, exist_ok=True)

        self.logs = root / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)

        self.sub = None
        self.ses = None

    def add_subject(self, sub_id):

        # Subject directory
        self.sub = self.data / f"sub-{sub_id}"
        self.sub.mkdir(parents=True, exist_ok=True)

    def add_session(self, ses_id):

        # Session directory
        self.ses = self.sub / f"ses-{ses_id}"
        self.ses.mkdir(parents=True, exist_ok=True)
