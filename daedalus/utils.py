# -*- coding: utf-8 -*-
# ==================================================================================================== #
#
#
#                    SCRIPT: utils.py
#
#
#          DESCRIPTION: Utility functions
#
#
#                       RULE: DAYW
#
#
#
#                  CREATOR: Sharif Saleki
#                         TIME: 05-26-2024-7810598105114117
#                       SPACE: Dartmouth College, Hanover, NH
#
# ==================================================================================================== #
import inspect
import yaml
import json
from pathlib import Path

import matplotlib.font_manager as fm
import seaborn as sns

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


def get_hypotenuse(point_a, point_b):
    """
    Calculate the Euclidean distance between two points in a 2D plane.

    Args:
        point_a (tuple): The coordinates of the first point.
        point_b (tuple): The coordinates of the second point.

    Returns:
        float: The Euclidean distance between the two points.
    """
    ax, ay = point_a
    bx, by = point_b

    return np.hypot(np.fabs(ax - bx), np.fabs(ay - by))


def get_caller():
    """
    Get the name of the function that called the function that's calling this function!

    Returns:
        str: The name of the calling function.
    """
    return inspect.currentframe().f_back.f_back.f_code.co_name


def set_plotting_style(theme_params, rc_params, font_dir=None):
    """
    Sets the plotting style for the analysis.

    Args:
        theme_params (dict): The parameters for the theme.
        rc_params (dict): The parameters for the rc (run command) settings.
        font_dir (str): The directory where the font is located.
    """
    if font_dir is not None:
        add_font(theme_params["font_name"], font_dir)
    sns.set_them(**theme_params, rc=rc_params)


def add_font(font_name: str, font_dir: str):
    """
    Finds the specified font for the plots and adds it to the font manager.

    Args:
        font_name (str): The name of the font to add.
        font_dir (str): The directory where the font is located.
    """
    font_files = list(Path(font_dir).glob(f"*{font_name}*"))
    for font_file in font_files:
        fm.fontManager.addfont(str(font_file))


def str2tuple(string):
    """
    Convert a string to a tuple.

    Args:
        string (str): The string to convert.

    Returns:
        tuple: The converted tuple.
    """
    # Clean the string
    return tuple(map(int, string.replace(" ", "")[1:-1].split(",")))


def time_index_from_sum(time_point, frame_times):
    """
    Find the frame number for a given time by cumulatively summing the frame times.

    Args:
        time_point (float): The time point to find the frame number for.
        frame_times (array): The frame times.

    Returns:
        int: The frame number.
    """
    return np.argmax(np.cumsum(frame_times) > time_point)


def time_index(time_point, frame_times):
    """
    Find the frame number for a given time.

    Args:
        time_point (float): The time point to find the frame number for.
        frame_times (array): The frame times.

    Returns:
        int: The frame number.
    """
    return np.argmax(frame_times > time_point)


def moving_average(x, w=30):
    avg = np.convolve(x, np.ones(w), 'valid') / w
    return np.concatenate((np.full(w-1, np.nan), avg))


def rotate_point(x, y, theta):
    """
    Rotates the x, y coordinates by an angle theta.

    Args:
        x (float): The x coordinate.
        y (float): The y coordinate.
        theta (float): The angle of rotation.

    Returns:
        tuple: The rotated x, y coordinates.
    """
    x_rot = x * np.cos(theta) - y * np.sin(theta)
    y_rot = x * np.sin(theta) + y * np.cos(theta)
    return x_rot, y_rot
