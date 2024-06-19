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
            "con": 0,
            "config": 10,
            "calib": 20,
            "rec": 30,
            "file": 40,


            "reset": 70,
            "idle": 80,
            "drift": 90,

            "exp": 100,
            "ses": 110,
            "block": 120,
            "trial": 130,
            "fix": 140,
            "stim": 150,
            "resp": 160,
            "fb": 170,

            "sys": 1000,
        }
        self.state_codes = {
            "init": 0,
            "ok": 1,
            "fail": 2,
            "lost": 3,
            "stop": 4,
            "fin": 5,
            "term": 6,
            "good": 7,
            "bad": 8,
            "timeout": 9
        }
        self.proc_names = {
            "con": "CONNECTION",
            "config": "CONFIGURATION",
            "calib": "CALIBRATION",
            "rec": "RECORDING",
            "file": "FILE",
            "reset": "RESET",
            "idle": "IDLE",
            "drift": "DriftCorrection",

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
            "timeout": "TIMEOUT"
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
