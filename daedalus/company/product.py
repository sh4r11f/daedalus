#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =================================================================================================== #
#
#
#                        SCRIPT: product.py
#
#
#               DESCRIPTION: A project 
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
from .managers import (
    FileManager, DirectoryManager, SettingsManager,
    DataManager, SubjectManager, SessionManager
)
from daedalus.log_handler import DaedalusLogger


class Study:
    """
    Project class to handle basic operations for a study

    Args:
        name (str): The name of the project
        root (str): The root directory for the project
        platform (str): The platform used for the project
        subject (int): The subject ID
        session (int): The session ID
        **kwargs: Additional keyword arguments

    Attributes:
        name (str): The name of the project
        root (str): The root directory for the project
        platform (str): The platform used for the project
        sub (SubjectManager): Subject manager
        ses (SessionManager): Session manager
        folders (DirectoryManager): Directory manager
        files (FileManager): File manager
        settings (SettingsManager): Settings manager
        version (str): The version of the study
        sett (SettingsManager): Alias for settings
        ver (str): Alias for version
        data (DataManager): Data manager
    """
    def __init__(self, name, root, platform, debug=False):

        # Setup
        self.name = name
        self.root = Path(root)
        self.platform = platform
        self.debug = debug

        # Subject and session manager
        self.sub_lord = SubjectManager()
        self.ses_lord = SessionManager()

        # Folders and files
        self.folder_lord = DirectoryManager(home=self.root, config=self.root / "config")
        config_files = {f.stem: f for f in self.folder_lord.config.glob("*.yaml")}
        self.file_lord = FileManager(**config_files)

        # Settings
        settings = utils.read_config(self.file_lord.settings)
        self.setting_lord = SettingsManager(
            study=settings["Study"],
            platform=settings["Platforms"][self.platform]
            )
        for f, path in config_files.items():
            self.setting_lord.load_from_file(f, path)
        self.version = f"v{self.setting_lord.study['Version']}"

        # Add folders from the settings file
        self.folder_lord.add(
            data=self.setting_lord.platform["Directories"].get("data"),
            tools=self.setting_lord.platform["Directories"].get("tools"),
            )

        # Data
        self.data_lord = DataManager()

    @property
    def folders(self):
        return self.folder_lord

    @property
    def files(self):
        return self.file_lord

    @property
    def sub(self):
        return self.sub_lord

    @property
    def ses(self):
        return self.ses_lord

    @property
    def sett(self):
        return self.setting_lord

    @property
    def data(self):
        return self.data_lord

    def __repr__(self):
        return f"{self.name} ({self.platform})"
