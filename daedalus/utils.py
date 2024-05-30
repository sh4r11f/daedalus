# -*- coding: utf-8 -*-
# ==================================================================================================== #
#                                                                                                                                                                                                      #
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
import numpy as np


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

import numpy as np

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