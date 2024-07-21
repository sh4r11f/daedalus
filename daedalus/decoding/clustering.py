#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =================================================================================================== #
#
#
#                        SCRIPT: clustering.py
#
#
#                   DESCRIPTION: Clustering algorithms
#
#
#                          RULE: DAYW
#
#
#
#                       CREATOR: Sharif Saleki
#                          TIME: 07-20-2024-7810598105114117
#                         SPACE: Dartmouth College, Hanover, NH
#
# =================================================================================================== #
from pathlib import Path
import os

import numpy as np
import pandas as pd

from sklearn.manifold import TSNE

from .base import ClusterDuck


class Teesni(ClusterDuck):
    def __init__(self, **kwargs):
        super().__init__(model="TSNE", **kwargs)

        self.manifold = TSNE(
            n_components=self.params["n_components"],
            preplexity=self.params["perplexity"],
            verbose=self.params["verbose"],
            random_state=self.params["random_state"],
            n_jobs=self.params["n_jobs"],
            angle=self.params["angle"],
            )

    def fit(self):
        
