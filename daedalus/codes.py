#!/usr/bin/env python3
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

        self.events = {
            "con": 0,
            "config": 10,
            "calib": 20,
            "rec": 30,
            "dl": 40,
            "edf": 50,
            "end": 60,
            "reset": 70,
            "idle": 80,

            "fix": 100
        }
        self.states = {
            "init": 0,
            "ok": 1,
            "fail": 2,
            "lost": 3,
            "stop": 4,
            "redo": 5,
            "term": 6,
            "valid": 7,
            "bad": 8,
        }
        self.names = {
            "con": "CONNECTION",
            "config": "CONFIGURATION",
            "calib": "CALIBRATION",
            "rec": "RECORDING",
            "dl": "DOWNLOAD",
            "edf": "EDF",
            "end": "END",
            "reset": "RESET",
            "idle": "IDLE",

            "fix": "FIXATION",

            "init": "INIT",
            "ok": "OK",
            "fail": "FAIL",
            "lost": "LOST",
            "stop": "STOP",
            "redo": "REDO",
            "term": "TERMINATED",
            "valid": "VALID",
            "bad": "INVALID"
        }

    def code(self, event, state):
        if event not in self.events.keys():
            raise ValueError(f"Event {event} not found in the codex")
        if state not in self.states.keys():
            raise ValueError(f"State {state} not found in the codex")
        return self.events[event] + self.states[state]

    def message(self, event, state):
        if event not in self.events.keys():
            raise ValueError(f"Event {event} not found in the codex")
        if state not in self.states.keys():
            raise ValueError(f"State {state} not found in the codex")
        return f"{event.upper()}_{state.upper()}"
