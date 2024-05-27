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
import sys

import pylink
from . import EyeLinkCoreGraphicsPsychoPy


class Eyelink:
	"""
	Class for controlling Eyelink 1000 eyetracker
	"""
	def __init__(self, trakcer_config, exp_name, exp_window, dummy):

  		# Setup
		self.tracker_config = trakcer_config
		self.exp_name = exp_name
		self.exp_window = exp_window
		self.dummy = dummy

		# Initialize the Eyelink tracker
		self.tracker = pylink.getEYELINK()

	def connect(self):
		"""
		Connect to the Eyelink tracker
		"""
		if self.dummy:
			self.tracker = pylink.EyeLink(None)
		else:
			try:
				self.tracker.open(self.tracker_config["address"])
			except RuntimeError as e:
				self.close()
				raise RuntimeError("Could not connect to the Eyelink tracker", e)

	def close(self):
		"""
		Close the connection to the Eyelink tracker
		"""
		if self.tracker.isConnected():
			self.tracker.close()
		sys.exit()

	def open_file(self):
		"""
		Open a file to record the data
		"""
		try:
			self.tracker.openDataFile(self.tracker_config["host_file"])
			# Add a header text to the EDF file to identify the current experiment name
			self.tracker.sendCommand(f"add_file_preamble_text {self.exp_name}")
		except RuntimeError as e:
			self.close()
			raise RuntimeError("Could not open a file to record the data", e)

	def get_tracker_version(self):
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
  
		# what eye events to save in the EDF file and to make available over the link
		self.tracker.sendCommand(f"file_event_filter = {self.tracker_config['file_event_filter']}")
		self.tracker.sendCommand(f"link_event_filter = {self.tracker_config['link_event_filter']}")
		# what sample data to save in the EDF data file and to make available over the link
		self.tracker.sendCommand(f"file_sample_data = {self.tracker_config['file_sample_data']}")
		self.tracker.sendCommand(f"link_sample_data = {self.tracker_config['link_sample_data']}")
		# set the calibration type and sample rate
		self.tracker.sendCommand(f"calibration_type = {self.tracker_config['calibration_type']}")
		self.tracker.sendCommand(f"sample_rate {self.tracker_config['sample_rate']}")
		# pass the display pixel coordinates (left, top, right, bottom) to the tracker
		self.tracker.sendCommand(f"screen_pixel_coords = {self.tracker_config['screen_pixel_coords']}")
		# write a DISPLAY_COORDS message to the EDF file. data Viewer needs this piece of info for proper visualization
		self.tracker.sendMessage(f"DISPLAY_COORDS = {self.tracker_config['display_coords']}")

	def calibrate(self):
		"""
		Calibrate the Eyelink 1000
		"""
		try:
			# Start the calibration
			self.tracker.doTrackerSetup()
			self.tracker.sendMessage("tracker_calibrated")
		except RuntimeError as e:
			self.tracker.exitCalibration()
			self.close()
			raise RuntimeError("Could not calibrate the Eyelink tracker", e)
