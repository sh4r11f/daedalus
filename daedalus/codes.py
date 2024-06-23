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
            "rew": "REWARD"
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
            "period": "PERIOD",
        }

    def code(self, proc, state):

        if proc not in self.proc_codes.keys():
            raise ValueError(f"Process {proc} not found in the codex")
        if state not in self.state_codes.keys():
            raise ValueError(f"State {state} not found in the codex")

        return self.proc_codes[proc] + self.state_codes[state]

    def message(self, proc, state):

        if proc not in self.proc_names.keys():
            raise ValueError(f"Process {proc} not found in the codex")
        if state not in self.state_names.keys():
            raise ValueError(f"State {state} not found in the codex")

        return f"{self.proc_names[proc]}_{self.state_names[state]}"

    def code2msg(self, code):
        """
        Convert a code to a message

        Args:
            code (float): The code to convert

        Returns:
            str: The message corresponding to the code
        """
        code_proc, code_state = modf(code)
        code_proc = int(code_proc)
        proc = [k for k, v in self.proc_codes.items() if v == code_proc][0]
        state = [k for k, v in self.state_codes.items() if v == code_state][0]

        return self.message(proc, state)

    def get_proc_name(self, code):
        """
        Get the process name from a code

        Args:
            code (float): The code to convert

        Returns:
            str: The process name corresponding to the code
        """
        code_proc, code_state = modf(code)
        code_proc = int(code_proc)
        proc_key = [k for k, v in self.proc_codes.items() if v == code_proc][0]
        proc_name = self.proc_names[proc_key].lower().capitalize()

        return proc_name
