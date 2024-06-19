# -*- coding: utf-8 -*-
# ==================================================================================================== #
#
#                                                                                                                                                                                                      #
#                    SCRIPT: utils.py                                                                                                                                                                 #
#                                                                                                                                                                                                      #
#                                                                                                                                                                                                      #
#          DESCRIPTION: Utility functions                                                                                                                                                                 #
#                                                                                                                                                                                                      #
#                                                                                                                                                                                                      #
#                       RULE: DAYW                                                                                                                                                                 #
#                                                                                                                                                                                                      #
#                                                                                                                                                                                                      #
#                                                                                                                                                                                                      #
#                  CREATOR: Sharif Saleki                                                                                                                                                #
#                         TIME: 05-26-2024-7810598105114117                                             #
#                       SPACE: Dartmouth College, Hanover, NH                                                                                                               #
#                                                                                                                                                                                                      #
# ==================================================================================================== #
import inspect
import yaml
import json
from pathlib import Path

import numpy as np


def read_config(file_path):
    """
    Reads and saves parameters specified in a json file under the config directory, e.g. config/params.json

    Args:
        file_path (str): The path to the .json or .yaml file.

    Returns:
        dict: The .json file is returned as a python dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not .json or .yaml.
    """
    # Find the parameters file
    params_file = Path(file_path)

    # Check the extension
    if params_file.suffix == ".json":
        # Check its status and read it
        if params_file.is_file():
            with open(params_file) as pf:
                params = json.load(pf)
                return params
        else:
            raise FileNotFoundError(f"{str(params_file)} does not exist")
    elif params_file.suffix == ".yaml":
        # Check its status and read it
        if params_file.is_file():
            with open(params_file) as pf:
                params = yaml.safe_load(pf)
                return params
        else:
            raise FileNotFoundError(f"{str(params_file)} does not exist")
    else:
        raise ValueError(f"Invalid file extension: {params_file.suffix}")


def find_in_configs(dicts: list, key: str, value: str):
    """
    Finds the value of a target in a list of dictionaries.

    Args:
        dicts (list): A list of dictionaries to search.
        key (str): The key that should match.
        value (str): The value for key .

    Returns:
        dict: The dictionary that contains the target value.
    """
    for d in dicts:
        if d[key] == value:
            return d


def get_hypot(orig_x, orig_y, end_x, end_y):
    """
    Calculate the Euclidean distance between two points in a 2D plane.

    Parameters:
        orig_x (float): The x-coordinate of the original point.
        orig_y (float): The y-coordinate of the original point.
        end_x (float): The x-coordinate of the end point.
        end_y (float): The y-coordinate of the end point.

    Returns:
        float: The Euclidean distance between the two points.
    """
    xdiff = np.fabs(orig_x - end_x)
    ydiff = np.fabs(orig_y - end_y)

    return np.hypot(xdiff, ydiff)


def get_caller():
    """
    Get the name of the function that called the current function.

    Returns:
        str: The name of the calling function.
    """
    return inspect.currentframe().f_back.f_code.co_name
