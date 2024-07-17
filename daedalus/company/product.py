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
from .managers import FileManager, DirectoryManager, SettingsManager, DataManager


class Project:
    def __init__(self, name, root, platform, **kwargs):

        self.name = name
        self.root = root
        self.platform = platform

        self.settings = SettingsManager(root, platform, **kwargs)
        self.version = self.settings.version

        self.folders = DirectoryManager()
        self.folders.add(
            config=self.root / "config",
            log=self.root / "log",
            figures=self.root / "figures",
            results=self.root / "results",
            scripts=self.root / "scripts",
            analysis=self.root / "analysis",
            data=self.settings.platform["Directories"].get("data"),
            tools=self.settings.platform["Directories"].get("tools"),
            fonts=self.settings.platform["Directories"].get("fonts"),
            )
        self.files = FileManager()
        self.data = DataManager()
