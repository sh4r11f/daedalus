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
from .managers import FileManager, DirectoryManager, SettingsManager


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
            data=self.root / "data",
            logs=self.root / "logs",
            raw=self.settings.platform["Directories"].get("raw"),
            tools=self.settings.platform["Directories"].get("tools"),
            fonts=self.settings.platform["Directories"].get("fonts"),
            )
        self.files = FileManager()
