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
    def __init__(self, name, root, platform, subject, session, debug=False):

        # Setup
        self.name = name
        self.root = Path(root)
        self.platform = platform
        self.debug = debug

        # Subject and session manager
        sub = int(subject)
        ses = int(session)
        self.sub = SubjectManager(id=sub, name=f"sub-{sub:02d}")
        self.ses = SessionManager(id=ses, name=f"ses-{ses:02d}")

        # Folders and files
        self.folders = DirectoryManager(home=self.root, config=self.root / "config")
        config_files = {f.stem: f for f in self.folders.config.glob("*.yaml")}
        self.files = FileManager(**config_files)

        # Settings
        settings = utils.read_config(self.files.settings)
        self.settings = SettingsManager(
            study=settings["Study"],
            platform=settings["Platforms"][self.platform]
            )
        for f, path in config_files.items():
            self.settings.load_from_file(f, path)
        self.version = f"v{self.settings.study['Version']}"

        # Add folders from the settings file
        self.folders.add(
            data=self.settings.platform["Directories"].get("data"),
            tools=self.settings.platform["Directories"].get("tools"),
            )

        # Data
        self.data = DataManager()

    @property
    def sett(self):
        return self.settings

    @property
    def ver(self):
        return self.version

    def __repr__(self):
        return f"{self.name} ({self.platform})"