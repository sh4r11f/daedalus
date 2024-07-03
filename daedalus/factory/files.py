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
import re


class FileManager:
    """
    FileManager class to handle file operations for the vision module

    Args:
        root (str): Root directory for the vision module

    Attributes:
        root (str): Root directory for the vision module
        data_dir (str): Data directory for the vision module
    """
    def __init__(self, name, root, version, debug):

        # Setup
        self.name = name
        self.root = Path(root)
        self.version = version
        self.debug = debug

        # Directories
        self.dirs = DirectoryManager(root, version)

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
        self.edf_display = None
        self.edf_host = None

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
        self.log = str(self.log)

    def _exist_rename(self, file):
        """
        Handle file exists error

        Args:
            file_path (str): The file path that already exists
        """
        if file.exists():
            try:
                file.rename(file.with_suffix(".BAK"))
            except FileExistsError:
                if self.debug:
                    file.unlink()
            return f"File {file.name} already exists. Renamed to {file.name}.BAK"

    def add_block(self, sub_id, ses_id, block_id, task_id):
        """
        Add a block to the FileManager object

        Args:
            sub_id (str): Subject ID
            ses_id (str): Session ID
            block_id (str): Block ID
            task_id (str): Task ID

        Returns:
            list: List of errors
        """
        errors = []
        pre_name = f"sub-{sub_id}_ses-{ses_id}_task-{task_id}_block-{block_id}"
        fname = pre_name + "_behavior.csv"
        behav_file = self.dirs.ses / fname
        errors.append(self._exist_rename(behav_file))

        fname = pre_name + "_stimuli.csv"
        stim_file = self.dirs.ses / fname
        errors.append(self._exist_rename(stim_file))
        
        fname = pre_name + "_frames.csv"
        frame_file = self.dirs.ses / fname
        errors.append(self._exist_rename(frame_file))
        
        fname = pre_name + "_EyeEvents.csv"
        eye_event_file = self.dirs.ses / fname
        errors.append(self._exist_rename(eye_event_file))
        
        fname = pre_name + "_EyeSamples.csv"
        eye_sample_file = self.dirs.ses / fname
        errors.append(self._exist_rename(eye_sample_file))
        
        fname = pre_name + "_EDFDisplay.csv"
        edf_display_file = self.dirs.ses / fname
        errors.append(self._exist_rename(edf_display_file))

        edf_host_file = f"{sub_id}_{ses_id}_{block_id}.edf"

        self.behavior = str(behav_file)
        self.stim_data = str(stim_file)
        self.frames = str(frame_file)
        self.eye_events = str(eye_event_file)
        self.eye_samples = str(eye_sample_file)
        self.edf_display = str(edf_display_file)
        self.edf_host = str(edf_host_file)

        return [e for e in errors if e is not None]

    def remove_block_from_names(self):

        self.behavior = re.sub(r"_block-\d+", "", self.behavior)
        self.stim_data = re.sub(r"_block-\d+", "", self.stim_data)
        self.frames = re.sub(r"_block-\d+", "", self.frames)
        self.eye_events = re.sub(r"_block-\d+", "", self.eye_events)
        self.eye_samples = re.sub(r"_block-\d+", "", self.eye_samples)
        self.edf_display = re.sub(r"_block-\d+", "", self.edf_display)

    def convert_to_str(self):

        self.behavior = str(self.behavior)
        self.stim_data = str(self.stim_data)
        self.frames = str(self.frames)
        self.eye_events = str(self.eye_events)
        self.eye_samples = str(self.eye_samples)
        self.edf_display = str(self.edf_display)
        self.edf_host = str(self.edf_host)

    def load_stimuli(self):
        """
        Load stimulus files into the FileManager object
        """
        # Load stimulus files
        files = list(self.dirs.stimuli.glob("*.png")) + list(self.dirs.stimuli.glob("*.jpg"))
        for file in files:
            setattr(self, str(file.stem), str(file))

    def get_file(self, file_name):
        """
        Get the file path for a given file name

        Args:
            file_name (str): The name of the file to get

        Returns:
            str: The file path
        """
        for attr in dir(self):
            if file_name in attr:
                return getattr(self, file_name)


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
