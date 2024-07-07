# -*- coding: utf-8 -*-
# ======================================================================================== #
#
#
#                    SCRIPT: codes.py
#
#
#          DESCRIPTION: Constants and codes for the vision module
#
#
#                       RULE: DAYW
#
#
#
#                  CREATOR: Sharif Saleki
#                         TIME: 06-12-2024-7810598105114117
#                       SPACE: Dartmouth College, Hanover, NH
#
# ======================================================================================== #
from math import modf


class Codex:

    def __init__(self):

        self.proc_codes = {
            "break": 0,
            "con": 1,
            "config": 2,
            "calib": 3,
            "rec": 4,
            "file": 5,
            "reset": 6,
            "idle": 7,
            "drift": 8,
            "rate": 9,
            "tracker": 30,
            "eye": 31,
            "edf": 32,

            "exp": 10,
            "ses": 11,
            "block": 12,
            "trial": 13,
            "fix": 14,
            "stim": 15,
            "resp": 16,
            "fb": 17,
            "cue": 18,
            "view": 19,
            "sys": 20,
            "iti": 21,
            "usr": 22,
            "sacc": 23,
            "rew": 24,
            "sub": 25,
            "smp": 26,
            "mask": 27,
            "data": 28,
        }
        self.state_codes = {
            "init": 0.0,
            "ok": 0.1,
            "fail": 0.2,
            "lost": 0.3,
            "stop": 0.4,
            "fin": 0.5,
            "term": 0.6,
            "good": 0.7,
            "bad": 0.8,
            "timeout": 0.9,
            "rep": 0.01,
            "maxout": 0.02,
            "done": 0.03,
            "onset": 0.04,
            "offset": 0.05,
            "per": 0.06,
            "null": 0.07,
            "dup": 0.08,
        }
        self.proc_names = {
            "con": "CONNECTION",
            "config": "CONFIGURATION",
            "calib": "CALIBRATION",
            "rec": "RECORDING",
            "file": "FILE",
            "reset": "RESET",
            "idle": "IDLE",
            "drift": "DRIFT_CORRECTION",
            "rate": "RATE",
            "tracker": "TRACKER",
            "eye": "EYE",
            "edf": "EDF",
            "exp": "EXPERIMENT",
            "ses": "SESSION",
            "block": "BLOCK",
            "trial": "TRIAL",
            "fix": "FIXATION",
            "stim": "STIMULUS",
            "resp": "RESPONSE",
            "fb": "FEEDBACK",
            "cue": "CUE",
            "view": "VIEW",
            "sys": "SYSTEM",
            "iti": "INTER_TRIAL_INTERVAL",
            "usr": "USER",
            "sacc": "SACCADE",
            "rew": "REWARD",
            "sub": "SUBJECT",
            "smp": "SAMPLE",
            "mask": "MASK",
            "data": "DATA",
        }
        self.state_names = {
            "init": "START",
            "ok": "OK",
            "fail": "FAILED",
            "lost": "LOST",
            "stop": "STOP",
            "fin": "FINISHED",
            "term": "TERMINATED",
            "good": "VALID",
            "bad": "INVALID",
            "timeout": "TIMEOUT",
            "rep": "REPEAT",
            "maxout": "MAXED_OUT",
            "done": "DONE",
            "onset": "ONSET",
            "offset": "OFFSET",
            "per": "PERIOD",
            "null": "NULL",
            "dup": "DUPLICATE"
        }

    def code(self, proc, state):

        if proc not in self.proc_codes.keys():
            raise ValueError(f"Process {proc} not found in the codex")
        if state not in self.state_codes.keys():
            raise ValueError(f"State {state} not found in the codex")

        return self.proc_codes[proc] + self.state_codes[state]

    def message(self, proc_key, state_key):

        if proc_key not in self.proc_names.keys():
            raise ValueError(f"Process {proc_key} not found in the codex")
        if state_key not in self.state_names.keys():
            raise ValueError(f"State {state_key} not found in the codex")

        return f"{self.proc_names[proc_key]}_{self.state_names[state_key]}"

    def code2msg(self, code):
        """
        Convert a code to a message

        Args:
            code (float): The code to convert

        Returns:
            str: The message corresponding to the code
        """
        return self.message(self.get_proc_key(code), self.get_state_key(code))

    def get_proc_key(self, code):
        """
        Get the process name from a code

        Args:
            code (float): The code to convert

        Returns:
            str: The process name corresponding to the code
        """
        proc_code = int(modf(code)[1])
        return [k for k, v in self.proc_codes.items() if v == proc_code][0]

    def get_state_key(self, code):
        """
        Get the state name from a code

        Args:
            code (float): The code to convert

        Returns:
            str: The state name corresponding to the code
        """
        state_code = round(modf(code)[0], 3)
        return [k for k, v in self.state_codes.items() if v == state_code][0]

    def get_proc_name(self, code):
        """
        Get the process name from a code

        Args:
            code (float): The code to convert

        Returns:
            str: The process name corresponding to the code
        """
        proc = self.get_proc_key(code)
        assert proc in self.proc_names.keys(), f"Process {proc} not found in the codex"
        return self.proc_names[proc]

    def get_state_name(self, code):
        """
        Get the state name from a code

        Args:
            code (float): The code to convert

        Returns:
            str: The state name corresponding to the code
        """
        state = self.get_state_key(code)
        assert state in self.state_names.keys(), f"State {state} not found in the codex"
        return self.state_names[state]