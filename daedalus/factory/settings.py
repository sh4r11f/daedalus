#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =================================================================================================== #
#
#
#                    SCRIPT: settings.py
#
#
#               DESCRIPTION: Settings manager
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
import numpy as np

from daedalus.utils import read_config


class SettingsManager:
    """
    Class to handle settings and parameters

    Args:
        config_dir (str): Directory for the configuration files
        version (str): Version of the module
        platform (str): Platform for the module
    """
    def __init__(self, config_dir, version, platform):

        # Setup
        self.config_dir = config_dir
        self.version = version

        settings = read_config(self.config_dir / "settings.yaml")
        self.study = settings["Study"]
        self.platform = settings["Platforms"][platform]

        config_files = {
            "analysis": "analysis.yaml",
            "exp": "experiment.yaml",
            "schedules": "schedules.yaml",
            "stimuli": "stimuli.yaml"
        }

        for attr, filename in config_files.items():
            file_path = self.config_dir / filename
            if file_path.exists():
                setattr(self, attr, read_config(file_path))
            else:
                setattr(self, attr, None)

        mon_file = self.config_dir / "monitors.yaml"
        mon_name = self.platform["Monitor"]
        if mon_file.exists():
            monitors = read_config(mon_file)
            self.monitor = monitors.get(mon_name, None)
        else:
            self.monitor = None

        gamma_file = self.config_dir / f"{mon_name}_gamma_grid.npy"
        if gamma_file.exists():
            self.gamma = np.load(str(gamma_file))
        else:
            self.gamma = None

        tracker_file = self.config_dir / "tracker.yaml"
        tracker_name = self.platform["Tracker"]
        if tracker_file.exists():
            trackers = read_config(tracker_file)
            self.tracker = trackers.get(tracker_name, None)
        else:
            self.tracker = None