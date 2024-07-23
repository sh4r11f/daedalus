#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =================================================================================================== #
#
#
#                        SCRIPT: base.py
#
#
#                   DESCRIPTION: Base statistician
#
#
#                          RULE: DAYW
#
#
#
#                       CREATOR: Sharif Saleki
#                          TIME: 07-21-2024-7810598105114117
#                         SPACE: Dartmouth College, Hanover, NH
#
# =================================================================================================== #
from pathlib import Path
import os

import numpy as np
import pandas as pd


class BaseStatistician:
    def __init__(self, data, params, **kwargs):
        self.data = data
        self.params = params
