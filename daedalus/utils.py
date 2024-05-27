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