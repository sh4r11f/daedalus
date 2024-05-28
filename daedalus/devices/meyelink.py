#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================================== #
#                                                                                                                                                                                                      #
#                                                                                                                                                                                                      #
#                    SCRIPT: eylink.py                                                                                                                                                                 #
#                                                                                                                                                                                                      #
#                                                                                                                                                                                                      #
#          DESCRIPTION: Class for controlling Eyelink tracker                                                                                                                                                                 #
#                                                                                                                                                                                                      #
#                                                                                                                                                                                                      #
#                       RULE: DAYW                                                                                                                                                                 #
#                                                                                                                                                                                                      #
#                                                                                                                                                                                                      #
#                                                                                                                                                                                                      #
#                  CREATOR: Sharif Saleki                                                                                                                                                #
#                         TIME: 05-27-2024-[78 105 98 105114117]                                             #
#                       SPACE: Dartmouth College, Hanover, NH                                                                                                               #
#                                                                                                                                                                                                      #
# ==================================================================================================== #
from psychopy.tools.monitorunittools import deg2pix

import pylink


class MeyeLink:
	"""
	Class for controlling Eyelink 1000 eyetracker
	"""
	def __init__(self, tracker_config, exp_name, exp_window, dummy):

  		# Setup
		self.tracker_config = tracker_config
		self.exp_name = exp_name
		self.exp_window = exp_window
		self.dummy = dummy

		self.tracker = None
		self.error = None

	def connect(self):
		"""
		Connect to the Eyelink tracker
		"""
		if self.dummy:
			self.tracker = pylink.EyeLink(None)
		else:
			try:
				self.tracker =  pylink.EyeLink(self.tracker_config["Connection"]["ip_address"])
			except RuntimeError as err:
				self.error = err
				return False
		return True

	def terminate(self, host_file, display_file):
		"""
		Close the connection to the Eyelink tracker
		"""
		msg = None
		if self.tracker.isConnected():

			if self.tracker.isRecording():
				pylink.pumpDelay(100)
				self.tracker.stopRecording()
    
			self.tracker.setOfflineMode()

            # Clear the Host PC screen and wait for 500 ms
			self.tracker.sendCommand('clear_screen 0')
			pylink.msecDelay(500)

            # Close the edf data file on the Host
			self.tracker.closeDataFile()

			# Download the EDF data file from the Host PC to the Display PC
			try:
				self.tracker.receiveDataFile(host_file, display_file)
			except RuntimeError as err:
				msg = err

			# Close the connection to the Eyelink tracker
			self.tracker.closeGraphics()
			self.tracker.close()
		else:
			msg = "Eyelink tracker is not connected"

		return msg

	def open_file(self, host_file):
		"""
		Open a file to record the data and initialize it
		"""
		try:
			self.tracker.openDataFile(host_file)
			# Add a header text to the EDF file to identify the current experiment name
			self.tracker.sendCommand(f"add_file_preamble_text {self.exp_name}")
			return True
		except RuntimeError as err:
			self.error = err
			return False

	def get_version(self):
		"""
		Get the version of the Eyelink tracker
		"""
		return self.tracker.getTrackerVersion()

	def configure(self):
		"""
		Configure the Eyelink 1000 connection to track the specified events and have the appropriate setup
		e.g., sample rate and calibration type.
		"""
		# Set the tracker parameters
		self.tracker.setOfflineMode()

		# Set the configuration from the config dict (which is a dictionary of dictionaries	)
		for config_type, config_dict in self.tracker_config.items():
			if config_type not in ["Connection", "Extra"]:
				for key, value in config_dict.items():
					self.tracker.sendCommand(f"{key} = {value}")
				
		# Rest of the setup
		win_width_idx= self.exp_window.size[0] - 1
		win_height_idx = self.exp_window.size[1] - 1
		self.tracker.sendCommnad(f"screen_pixel_coords = 0 0 {win_width_idx} {win_height_idx}")
		self.tracker.sendMessage(f"DISPLAY_COORDS = {self.tracker_config['display_coords']}")

	def calibrate(self):
		"""
		Calibrate the Eyelink 1000
		"""
		try:
			# Start the calibration
			self.tracker.doTrackerSetup()
			self.tracker.sendMessage("tracker_calibrated")
			return True
		except RuntimeError as err:
			self.tracker.exitCalibration()
			self.error = err
			return False

	def check_eye(self):
		"""
		Check if the eye is being tracked
		"""
		used = self.tracker.eyeAvailable()
		intended = self.tracker_config["Extra"]["active_eye"]
		if used == 0 and intended == "LEFT":
			self.tracker.sendMessage("EYE_USED 0 LEFT")
		elif used == 1 and intended == "RIGHT":
			self.tracker.sendMessage("EYE_USED 1 RIGHT")
		elif used == 2 and intended == "BINOCULAR":
			self.tracker.sendMessage("EYE_USED 2 BINOCULAR")
		else:
			return False
		return True