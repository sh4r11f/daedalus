#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================================== #
#
#
#                    SCRIPT: config.py
#
#
#          DESCRIPTION: Class for configuration parameters and settings
#
#
#                       RULE: DAYW
#
#
#
#                  CREATOR: Sharif Saleki
#                         TIME: 06-07-2024-7810598105114117
#                       SPACE: Dartmouth College, Hanover, NH
#
# ======================================================================================== #
from pathlib import Path

from daedalus.utils import read_config


class Configuration:
    """
    Class for configuration parameters and settings.
    """
    def __init__(self, name, config_dir):
        """
        Initializes the configuration object.

        Args:
            config_file (str): The path to the configuration file.
        """
        self.name = name
        self.config_dir = config_dir
        self.settings = None
        self.parameters = None

    def find_yaml_files(self):
        """
        Finds all the .yaml files in the config directory.

        Returns:
            list: A list of all the .yaml files in the config directory.
        """
        config_path = Path(self.config_dir)
        yaml_files = list(config_path.glob("*.yaml"))
        return yaml_files

    def read_yaml_files(self):
        """
        Reads and saves parameters specified in .yaml files under the config directory.

        Returns:
            dict: A dictionary containing the parameters from all the .yaml files in the config directory.
        """
        yaml_files = self.find_yaml_files()
        config = {}
        for yaml_file in yaml_files:
            config.update(read_config(yaml_file))
        return config