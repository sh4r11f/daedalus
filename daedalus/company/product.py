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
    """
    def __init__(self, name, root, platform, subject, session, **kwargs):

        # Setup
        self.name = name
        self.root = root
        self.platform = platform

        # Subject and session manager
        self.sub = SubjectManager(id=int(subject), name=f"sub-{int(subject):02d}")
        self.ses = SessionManager(id=(session), name=f"ses-{int(session):02d}")

        # Folders and files
        self.folders = DirectoryManager(
            config=self.root / "config",
            scripts=self.root / "scripts",
            analysis=self.root / "analysis",
            )
        config_files = {f.stem: f for f in self.folders.config.glob("*.yaml")}
        self.files = FileManager(**config_files)

        # Settings
        settings = utils.read_config(self.files.settings)
        self.sett = SettingsManager(platform, **settings)
        self.ver = f"v{self.sett.study['Version']}"

        for f, path in config_files.items():
            self.sett.load(f, path)

        self.folders.add(
            data=self.sett.platform["Directories"].get("data"),
            tools=self.sett.platform["Directories"].get("tools"),
            fonts=self.sett.platform["Directories"].get("fonts"),
            )
        self.folders.make(
            log=self.root / "log" / self.ver,
            figures=self.root / "figures" / self.ver,
            results=self.root / "results" / self.ver,
            )

        # Data
        self.data = DataManager()
