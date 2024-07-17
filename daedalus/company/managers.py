#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =================================================================================================== #
#
#
#                        SCRIPT: managers.py
#
#
#               DESCRIPTION: Manage tasks and stuff
#
#
#                           RULE: DAYW
#
#
#
#                      CREATOR: Sharif Saleki
#                            TIME: 07-16-2024-7810598105114117
#                          SPACE: Dartmouth College, Hanover, NH
#
# =================================================================================================== #
from pathlib import Path
from daedalus import utils


class BaseManager:
    """
    BaseManager class to handle base operations for the vision module
    """
    def __init__(self, **kwargs):
        self.name = kwargs.get("name")

    def add(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    def get(self, name):
        """
        Get the object for a given name

        Args:
            name (str): The name of the object to get

        Returns:
            object: The object
        """
        for attr in dir(self):
            if name in attr:
                return getattr(self, name)


class FileManager(BaseManager):
    """
    FileManager class to handle file operations
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _make_backup(self, file):
        """
        Handle file exists error

        Args:
            file_path (str): The file path that already exists
        """
        backup = file.with_suffix(".BAK")
        if backup.exists():
            backup.unlink()
        file.rename(backup)

    def add(self, **kwargs):
        for key, val in kwargs.items():
            val = Path(val)
            if val.exists():
                self._make_backup(val)
            setattr(self, key, val)


class DirectoryManager(BaseManager):
    """
    DirectoryManager class to handle directory operations for the vision module
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def add(self, **kwargs):
        for key, val in kwargs.items():
            val = Path(val)
            if not val.exists():
                val.mkdir(parents=True, exist_ok=True)
            setattr(self, key, val)


class SettingsManager:
    """
    Class to handle settings and parameters

    Args:
        config_dir (str): Directory for the configuration files
        version (str): Version of the module
        platform (str): Platform for the module
    """
    def __init__(self, root, platform, project_key="Study"):

        # Setup
        self.root = root
        self.config_dir = self.root / "config"

        settings = utils.read_config(self.config_dir / "settings.yaml")
        self.settings = settings
        self.main = settings[project_key]
        self.platform = settings["Platforms"][platform]

        self.version = self.main["Version"]

    def load_config(self, name):
        """
        Load a configuration file

        Args:
            config_file (str): The configuration file to load
        """
        return utils.read_config(self.config_dir / f"{name}.yaml")

    def add(self, *args):
        """
        Add a configuration file to the settings manager
        """
        for name in args:
            setattr(self, name, self.load_config(name))


class DataManager(BaseManager):
    """
    DataManager class to handle data operations
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class SessionManager(BaseManager):
    """
    SessionManager class to handle session operations
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.data = DataManager()


class SubjectManager(BaseManager):
    """
    SubjectManager class to handle subject operations
    """
    def __init__(self, sessions, **kwargs):
        super().__init__(**kwargs)

        for ses in sessions:
            setattr(self, ses, SessionManager())
