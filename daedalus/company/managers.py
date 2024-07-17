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


class FileManager:
    """
    FileManager class to handle file operations for the vision module

    Args:
        root (str): Root directory for the vision module

    Attributes:
        root (str): Root directory for the vision module
        data_dir (str): Data directory for the vision module
    """
    def __init__(self, **kwargs):
        self.name = kwargs.get("name")

    def add(self, **kwargs):

        for key, val in kwargs.items():
            setattr(self, key, val)

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

    def get_file(self, file_name):
        """
        Get the file path for a given file name

        Args:
            file_name (str): The name of the file to get

        Returns:
            str: The file path
        """
        if isinstance(file_name, Path):
            file_name = file_name.name

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
    def __init__(self, **kwargs):

        # Setup
        self.name = kwargs.get("name")

    def add(self, **kwargs):
        for key, val in kwargs.items():
            if isinstance(val, str):
                val = Path(val)
            if not val.exists():
                val.mkdir(parents=True, exist_ok=True)
            setattr(self, key, val)

    def get(self, dir_name):
        """
        Get the directory path for a given directory name

        Args:
            dir_name (str): The name of the directory to get

        Returns:
            str: The directory path
        """
        for attr in dir(self):
            if dir_name in attr:
                return str(getattr(self, dir_name))


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

    def load_config(self, config_name):
        """
        Load a configuration file

        Args:
            config_file (str): The configuration file to load
        """
        return utils.read_config(self.config_dir / f"{config_name}.yaml")

    def add_config(self, config_name):
        """
        Add a configuration file to the settings manager

        Args:
            config_name (str): The name of the configuration file
        """
        setattr(self, config_name, self.load_config(config_name))
