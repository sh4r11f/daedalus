#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =================================================================================================== #
#
#
#                        SCRIPT: product.py
#
#
#                   DESCRIPTION: Class for studies, projects, experiments, etc.
#
#
#                          RULE: DAYW
#
#
#
#                       CREATOR: Sharif Saleki
#                          TIME: 07-16-2024-7810598105114117
#                         SPACE: Dartmouth College, Hanover, NH
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
        debug (bool): Whether to enable debug mode

    Attributes:
        name (str): The name of the project
        root (str): The root directory for the project
        platform (str): The platform used for the project
        sub_lord (SubjectManager): Subject manager
        ses_lord (SessionManager): Session manager
        dir_lord (DirectoryManager): Directory manager
        file_lord (FileManager): File manager
        setting_lord (SettingsManager): Settings manager
        version (str): The version of the study
        data (DataManager): Data manager
    """
    def __init__(self, name, root, platform, debug=False):

        # Setup
        self.name = name
        self.root = Path(root)
        self.platform = platform
        self.debug = debug

        # Subject and session manager
        self.sub = SubjectManager()
        self.ses = SessionManager()

        # Folders
        self.folders = DirectoryManager(home=self.root, config=self.root / "config")

        # Files
        config_files = {f.stem: f for f in self.folders.config.glob("*.yaml")}  # type: ignore
        self.files = FileManager(**config_files)

        # Settings
        settings = utils.read_config(self.files.settings)  # type: ignore
        self.settings = SettingsManager(
            study=settings["Study"],
            platform=settings["Platforms"][self.platform]
            )
        for f, path in self.files.iter_files():
            self.settings.load_from_file(f, path)

        self.settings.add(version=self.settings.study['Version'])  # type: ignore
        self.settings.add(version_name=f"v{self.settings.version}")  # type: ignore

        # Data
        self.data = DataManager()
        self.folders.add(data=Path(self.settings.platform["Directories"].get("data")))  # type: ignore

        # Logger
        self.folders.add(log=self.root / "log" / self.settings.version_name)
        self.logger = None

    def make_logger(self, logger_name, file_name):
        """
        Create a logger

        Args:
            logger_name (str): The name of the logger
            file_name (str): The name of the log file
        """
        self.files.make(log=self.folders.log / f"{file_name}.log")  # type: ignore
        self.logger = DaedalusLogger(
            name=logger_name, log_file=self.files.log, debug_mode=self.debug  # type: ignore
            )

    @property
    def sett(self):
        return self.settings
