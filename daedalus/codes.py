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

            "exp": 10,
            "ses": 11,
            "block": 12,
            "trial": 13,
            "fix": 14,
            "stim": 15,
            "resp": 16,
            "fb": 17,

            "sys": 20,
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
            "exp": "EXPERIMENT",
            "ses": "SESSION",
            "block": "BLOCK",
            "trial": "TRIAL",
            "fix": "FIXATION",
            "stim": "STIMULUS",
            "resp": "RESPONSE",
            "fb": "FEEDBACK",
            "sys": "SYSTEM"
        }
        self.state_names = {
            "init": "START",
            "ok": "SUCCESS",
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

        code_proc = code // 10
        code_state = code % 10
        proc = [k for k, v in self.proc_codes.items() if v == code_proc][0]
        state = [k for k, v in self.state_codes.items() if v == code_state][0]

        return self.message(proc, state)
