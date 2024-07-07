#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================================== #
#
#
#                    SCRIPT: stimulus_factory.py
#
#
#          DESCRIPTION: Manufactures stimuli
#
#
#                       RULE: DAYW
#
#
#
#                  CREATOR: Sharif Saleki
#                         TIME: 05-27-2024-[78 105 98105114117]
#                       SPACE: Dartmouth College, Hanover, NH
#
# ==================================================================================================== #
from pathlib import Path
from psychopy import visual, monitors
from psychopy.tools.monitorunittools import deg2pix, pix2deg
from psychopy_visionscience import NoiseStim
from daedalus.utils import str2tuple


class StimulusFactory:
    """
    A class that creates various visual stimuli used in an experiment.

    Args:
        name (str): The name of the task.
        root (str): The directory path of the project.
        version (str): The version of the experiment.
        platform (str): The platform on which the experiment is running.
        monitor: The monitor object for stimulus presentation.
        window: The window object for displaying stimuli.

    Attributes:
        name (str): The name of the task.
        root (str): The directory path of the project.
        version (str): The version of the experiment.
        platform (str): The platform on which the experiment is running.
        monitor: The monitor object for stimulus presentation.
        window: The window object for displaying stimuli.
        settings (dict): The settings for the experiment.
        params (dict): The parameters for the stimuli.
    """
    def __init__(self, root, params, font_dir=None):

        self.root = root
        self.params = params

        if font_dir is None:
            self.font_flles = None
        else:
            self.font_files = list(Path(font_dir).glob("*.otf")) + list(Path(font_dir).glob("*.ttf"))
            self.font_files = [str(f) for f in self.font_files]

        self.monitor = None
        self.window = None

    def set_display(self, monitor, window):
        """
        Set the monitor and window objects for the stimulus factory.

        Args:
            monitor: The monitor object for stimulus presentation.
            window: The window object for displaying stimuli.
        """
        self.monitor = monitor
        self.window = window

    def make_message(self, name, text=None):
        """
        Creates a message stimulus.

        Args:
            name (str): The name of the stimulus.
            text (str): The message to display.

        Returns:
            visual.TextStim: The message stimulus.
        """
        params = self.params["Message"]
        width = self.window.size[0] / params["wrap_ratio"]
        if self.window.units == "deg":
            width = deg2pix(width, self.monitor)

        stim = visual.TextStim(
            win=self.window,
            text=text,
            font=params["font"],
            fontFiles=self.font_files,
            height=params["font_size"],
            color=str2tuple(params["normal_text_color"]),
            alignText='left',
            anchorHoriz='center',
            anchorVert='center',
            wrapWidth=width,
            autoLog=False
        )
        setattr(self, name, stim)
        return stim

    def make_countdown_timer(self, name):
        """
        Creates a countdown timer stimulus.

        Args:
            name (str): The name of the stimulus.

        Returns:
            visual.TextStim: The countdown timer stimulus.
        """
        params = self.params["Countdown"]
        stim = visual.TextStim(
            self.window,
            text="",
            font=params["font"],
            height=params["font_size"],
            color=str2tuple(params["color"]),
            autoLog=False
        )
        setattr(self, name, stim)
        return stim

    def make_instructions_image(self, name, image_file):
        """
        Make an instruction image stimulus.

        Args:
            file_name (str): The name of the image file.
            image_file (str): The path to the image file.

        Returns:
            visual.ImageStim: The instruction image stimulus.
        """
        scale = self.params["Instructions"]["image_scale"]
        width, height = self.window.size
        width = width * scale
        height = height * scale
        if self.window.units == "deg":
            width = deg2pix(width, self.monitor)
            height = deg2pix(height, self.monitor)
        stim = visual.ImageStim(
            self.window,
            name=name,
            image=image_file,
            mask=None,
            pos=(0, 0),
            size=(width, height),
            anchor="center",
            ori=0,
            color=(1, 1, 1),
            colorSpace="rgb",
            contrast=1.0,
            opacity=1.0,
            depth=0,
            autoLog=False
        )
        setattr(self, name, stim)

        return stim

    def make_instructions_text(self, name, text=None):
        """
        Make an instruction text stimulus.

        Args:
            name (str): The name of the stimulus.
            text (str): The text to display in the stimulus.

        Returns:
            visual.TextStim: The instruction text stimulus.
        """
        params = self.params["Instructions"]
        if text is None:
            text = params.get(["text"], None)

        # Create the text stimulus
        stim = visual.TextStim(
            self.window,
            name=name,
            text=text,
            font=params["font"],
            height=params["font_size"],
            fontFiles=self.font_flles,
            color=(1, 1, 1),
            colorSpace="rgb",
            opacity=1.0,
            autoLog=False
        )
        setattr(self, name, stim)
        return stim

    def make_fixation(self, name):
        """
        Creates a fixation stimulus.

        Args:
            name (str): The name of the stimulus.

        Returns:
            visual.Circle: The fixation stimulus.
        """
        params = self.params["Fixation"]
        size = params["size"]
        if self.window.units == "pix":
            size = deg2pix(size, self.monitor)
        stim = visual.Circle(
            self.window,
            name=name,
            size=size,
            pos=str2tuple(params["position"]),
            lineColor=str2tuple(params["line_color"]),
            fillColor=str2tuple(params["normal_color"]),
            autoLog=False
        )
        setattr(self, name, stim)
        return stim

    def make_visual_mask(self, name):
        """
        Creates a dynamic mask stimulus.

        Args:
            name (str): The name of the stimulus.

        Returns:
            visual.GratingStim: The dynamic mask stimulus.
        """
        params = self.params["Mask"]
        mask = NoiseStim(
            win=self.window,
            name=name,
            size=(params["size"], params["size"]),
            noiseType=params["type"],
            noiseElementSize=params["noise_size"],
            noiseBaseSf=params["spatial_frequency"],
            noiseBW=params["spatial_frequency_bandwidth"],
            noiseBWO=params["orientation_bandwidth"],
            noiseFilterLower=params["lower_cutoff"] / params["size"],
            noiseFilterUpper=params["upper_cutoff"] / params["size"],
            noiseFilterOrder=params["filter_order"],
            color=str2tuple(params["color"]),
            colorSpace='rgb',
            opacity=1,
            blendmode='avg',
            contrast=1.0,
            texRes=params["resolution"],
            noiseClip=params["clip_value"],
            interpolate=False
        )
        # size = 512
        # noise_size = 16
        # tex_res = 64
        # sf = 5
        # bw = 1
        # ori = 30
        # low = 5 / size
        # high = 20 / size
        # mask = NoiseStim(
        #     win=self.window,
        #     # mask="sqrXsqr",
        #     noiseType="Binary",  # Black and white noise
        #     noiseElementSize=noise_size,  # Size of noise elements
        #     noiseBaseSf=sf,  # Base spatial frequency
        #     noiseBW=bw,  # Spatial frequency bandwidth
        #     noiseBWO=ori,  # Orientation bandwidth (degrees)
        #     noiseFilterLower=low,  # Lower cutoff frequency for the filter
        #     noiseFilterUpper=high,  # Upper cutoff frequency for the filter
        #     noiseFilterOrder=1,  # Filter order
        #     # sf=sf,
        #     size=(size, size),  # Size of the noise patch
        #     color=(1, 1, 1),  # Color of the noise
        #     colorSpace='rgb',
        #     opacity=1,
        #     blendmode='avg',
        #     contrast=1.0,
        #     texRes=tex_res,  # Texture resolution
        #     noiseClip=3.0,  # Clipping value
        #     interpolate=False
        # )
        setattr(self, name, mask)
        return mask

    def make_single_drift(self, name):
        """
        Creates a single drift stimulus.

        Args:
            name (str): The name of the stimulus.

        Returns:
            visual.GratingStim: The single drift stimulus.
        """
        sd_params = self.params["SingleDrift"]
        size = sd_params["size"]
        sf = sd_params["spatial_frequency"]
        if self.window.units == "pix":
            size_px = deg2pix(size, self.monitor)
            sf_px = sf * pix2deg(1, self.monitor)  # Convert from cpd to cpp
        stim = visual.GratingStim(
            self.window,
            name=name,
            tex="sin",
            mask="gauss",
            # mask_params={"sd": sd_params["gaussian_sd"]},
            size=size_px,
            sf=sf_px,
            phase=sd_params["phase"],
            autoLog=False
        )
        setattr(self, name, stim)
        return stim

    def make_double_drift(self, name):
        """
        Creates a double drift stimulus.

        Args:
            name (str): The name of the stimulus.

        Returns:
            visual.GratingStim: The double drift stimulus.
        """
        params = self.params["DoubleDrift"]
        size = params["size"]
        sf = params["spatial_frequency"]
        if self.window.units == "pix":
            size = deg2pix(size, self.monitor)
            sf = sf * pix2deg(1, self.monitor)

        dd = visual.GratingStim(
            self.window,
            name=name,
            tex="sin",
            mask="gauss",
            # mask_params={"sd": dd_params["gaussian_sd"]},
            size=size,
            sf=sf,
            ori=params["orientation"],
            phase=params["phase"],
            contrast=params["contrast"],
            autoLog=False
        )
        setattr(self, name, dd)
        return dd
    
    def make_aperture(self, name):
        """
        Creates a hard aperture stimulus.

        Args:
            name (str): The name of the stimulus.

        Returns:
            visual.GratingStim: The hard aperture stimulus.
        """
        params = self.params["Aperture"]
        gabor_params = self.params["SingleDrift"]
        radius = params["radius_scale"] * gabor_params["size"]
        lw = params["line_width"]
        if self.window.units == "pix":
            radius = deg2pix(radius, self.monitor)
            lw = deg2pix(params["line_width"], self.monitor)

        stim = visual.Circle(
            self.window,
            name=name,
            radius=radius,
            contrast=params["contrast"],
            lineWidth=lw,
            autoLog=False
        )
        setattr(self, name, stim)
        return stim

    def make_cue_fixation(self, name):
        """
        Creates a cue stimulus.

        Args:
            name (str): The name of the stimulus.

        Returns:
            visual.Circle: The cue stimulus.
        """
        params = self.params["Fixation"]
        size = params["size"]
        if self.window.units == "pix":
            size = deg2pix(size, self.monitor)

        cue = visual.Circle(
            self.window,
            name=name,
            size=size,
            pos=str2tuple(params["position"]),
            lineColor=str2tuple(params["line_color"]),
            fillColor=str2tuple(params["cue_color"]),
            autoLog=False
        )
        setattr(self, name, cue)
        return cue

    def make_warn_fixation(self, name):
        """
        Creates a warning stimulus.

        Args:
            name (str): The name of the stimulus.

        Returns:
            visual.TextStim: The warning stimulus.
        """
        params = self.params["Fixation"]
        size = params["size"]
        if self.window.units == "pix":
            size = deg2pix(size, self.monitor)
        stim = visual.Circle(
            self.window,
            name=name,
            size=size,
            pos=str2tuple(params["position"]),
            lineColor=str2tuple(params["line_color"]),
            fillColor=str2tuple(params["warning_color"]),
            autoLog=False
        )
        setattr(self, name, stim)
        return stim

    def make_feedback_fixation(self, name):
        """
        Creates a feedback fixation stimulus.

        Args:
            name (str): The name of the stimulus.

        Returns:
            visual.TextStim: The good job stimulus.
        """
        params = self.params["Fixation"]
        size = params["size"]
        if self.window.units == "pix":
            size = deg2pix(size, self.monitor)

        stim = visual.Circle(
            self.window,
            name=name,
            size=size,
            pos=str2tuple(params["position"]),
            lineColor=str2tuple(params["line_color"]),
            fillColor=str2tuple(params["feedback_color"]),
            autoLog=False
        )
        setattr(self, name, stim)
        return stim

    def make_reward_bar(self, name):
        """
        Creates a reward bar stimulus.

        Args:
            name (str): The name of the stimulus.

        Returns:
            visual.Rect: The reward bar stimulus.
        """
        rewbar_params = self.params["Reward"]["Bar"]
        w, h = self.window.size
        height = rewbar_params["height"]
        if self.window.units == "pix":
            height = deg2pix(height, self.monitor)
        else:
            w = pix2deg(w, self.monitor)
            h = pix2deg(h, self.monitor)
        pos = (0, -h / 2 + height / 2)

        stim = visual.Rect(
            self.window,
            name=name,
            size=(w, height),
            pos=pos,
            lineWidth=rewbar_params["line_width"],
            fillColor=str2tuple(rewbar_params["fill_color"]),
            lineColor=str2tuple(rewbar_params["line_color"]),
            autoLog=False
        )
        setattr(self, name, stim)
        return stim

    def make_bar_filler(self, name):
        """
        Creates a bar filler stimulus.

        Args:
            name (str): The name of the stimulus.

        Returns:
            visual.Rect: The bar filler stimulus.
        """
        params = self.params["Reward"]["Filler"]
        win_w, win_h = self.window.size

        # Normalize
        height = params["height"]
        if self.window.units == "pix":
            height = deg2pix(height, self.monitor)
        else:
            win_w = pix2deg(win_w, self.monitor)
            win_h = pix2deg(win_h, self.monitor)
        position = (-win_w / 2, -win_h / 2 + height / 2)

        stim = visual.Rect(
            self.window,
            name=name,
            size=(0, height),
            pos=position,
            fillColor=str2tuple(params["fill_color"]),
            lineWidth=0,
            autoLog=False
        )
        setattr(self, name, stim)
        return stim

    def make_reward_image(self, name, stim_file):
        """
        Creates a feedback image stimulus.

        Args:
            name (str): The name of the stimulus.
            stim_file (str): The path to the image file.

        Returns:
            visual.ImageStim: The feedback image stimulus.
        """
        # Load the correct image file based on the correctness of the feedback
        params = self.params["Reward"]["Image"]

        # Convert size
        size = params["size"]
        if self.window.units == "pix":
            size = deg2pix(size, self.monitor)
        size = size * params["scale"]

        # Create the feedback image stimulus
        feedback_img = visual.ImageStim(
            self.window,
            name=name,
            image=stim_file,
            size=size,
            autoLog=False
        )
        setattr(self, name, feedback_img)
        return feedback_img

    def make_reward_text(self, name):
        """
        Creates a feedback text stimulus.

        Args:
            name (str): The name of the stimulus.

        Returns:
            visual.TextStim: The feedback text stimulus.
        """
        params = self.params["Reward"]["Text"]
        stim = visual.TextStim(
            self.window,
            name=name,
            font=params["font"],
            fontFiles=self.font_files,
            height=params["font_size"],
            color=str2tuple(params["color"]),
            autoLog=False
        )
        setattr(self, name, stim)
        return stim

    def make_debrief_image(self, name, image_file):
        """
        Make a debrief image stimulus.

        Args:
            name (str): The name of the stimulus.
            image_file (str): Path of the image file.

        Returns:
            visual.ImageStim: The debrief image stimulus.
        """
        scale = self.params["Debrief"]["image_scale"]
        width, height = self.window.size
        width = width * scale
        height = height * scale
        if self.window.units == "deg":
            width = deg2pix(width, self.monitor)
            height = deg2pix(height, self.monitor)
        pos = (width / 2, 0)

        stim = visual.ImageStim(
            win=self.window,
            name=name,
            image=image_file,
            mask=None,
            pos=pos,
            size=(width, height),
            anchor="left",
            ori=0,
            color=(1, 1, 1),
            colorSpace="rgb",
            contrast=1.0,
            opacity=1.0,
            depth=0,
            autoLog=False
        )
        setattr(self, name, stim)
        return stim

    def make_performance_image(self, name, plot_file=None):
        """
        Make a performance image stimulus.

        Args:
            name (str): The name of the stimulus.
            plot_file (str): The path to the plot file.

        Returns:
            visual.ImageStim: The performance image stimulus.
        """
        scale = self.params["Debrief"]["image_scale"]
        width, height = self.window.size
        width = width * scale
        height = height * scale
        if self.window.units == "deg":
            width = deg2pix(width, self.monitor)
            height = deg2pix(height, self.monitor)
        pos = (width / 2, 0)

        stim = visual.ImageStim(
            win=self.window,
            name=name,
            image=plot_file,
            mask=None,
            units="pix",
            pos=pos,
            size=(width, height),
            anchor="right",
            ori=0,
            color=(1, 1, 1),
            colorSpace="rgb",
            contrast=1.0,
            opacity=1.0,
            depth=0,
            autoLog=False
        )
        setattr(self, name, stim)
        return stim


class DisplayFactory:
    """
    A class that creates various visual displays used in an experiment.

    Args:
        monitor_name (str): The name of the monitor.
        monitor_params (dict): The parameters for the monitor.
        gamma (list): The gamma correction values.
        window_params (dict): The parameters for the window.
        debug (bool): The debug status of the experiment.

    Attributes:
        mon_name (str): The name of the monitor.
        mon_params (dict): The parameters for the monitor.
        gamma (list): The gamma correction values.
        win_params (dict): The parameters for the window.
        debug (bool): The debug status of the experiment.
        monitor: The monitor object for the display.
        window: The window object for the display.
    """
    def __init__(self, monitor_name, monitor_params, gamma, window_params):

        self.mon_name = monitor_name
        self.mon_params = monitor_params
        self.gamma = gamma
        self.win_params = window_params

        self.monitor = None
        self.window = None

    def start(self, debug=False):
        """
        Start the display factory.
        """
        self.monitor = self.make_monitor()
        if debug:
            self.window = self.debug_window(self.monitor)
        else:
            self.window = self.make_window(self.monitor)

    def make_monitor(self):
        """
        Creates a monitor object for the experiment.
        """
        available = monitors.getAllMonitors()
        if self.mon_name in available:
            monitor = monitors.Monitor(name=self.mon_name)
        else:
            # Make the monitor object and set width, distance, and resolution
            monitor = monitors.Monitor(
                name=self.mon_name,
                width=self.mon_params["size_cm"][0],
                distance=self.mon_params["distance"],
                autoLog=False
            )
            monitor.setSizePix(self.mon_params["size_pix"])

            # Gamma correction
            if self.gamma is not None:
                monitor.setLineariseMethod(1)  # (a + b*xx)**gamma
                monitor.setGammaGrid(self.gamma)
            else:
                monitor.setGamma(None)

            # Save for future use
            # monitor.save()

        return monitor

    def make_window(self, monitor):
        """
        Creates a window object for the experiment.
        """
        # Make the window object
        window = visual.Window(
            name='ExperimentWindow',
            monitor=monitor,
            fullscr=True,
            units=self.win_params["units"],
            size=self.mon_params["size_pix"],
            allowGUI=False,
            waitBlanking=True,
            color=self.win_params["background_color"],  # default to mid-grey
            screen=0,  # the internal display is used by default
            checkTiming=False,
            autoLog=False
        )
        window.mouseVisible = False

        return window

    def debug_window(self, monitor):
        """
        Create a debug window.
        """
        # Make the window object
        window = visual.Window(
            name='DebugWindow',
            monitor=monitor,
            fullscr=False,
            units=self.win_params["units"],
            size=[1200, 800],
            allowGUI=True,
            waitBlanking=True,
            color=self.win_params["background_color"],  # default to mid-grey
            screen=0,  # the internal display is used by default
            checkTiming=False,
            autoLog=False
        )
        window.mouseVisible = True

        return window

    def show(self, stim):
        """
        Show a stimulus on the display.
        """
        stim.draw()
        self.window.flip()

    def close(self):
        """
        Close the display.
        """
        self.window.close()

    def clear(self):
        """
        Clear the display.
        """
        # Reset background color
        self.window.color = str2tuple(self.win_params["background_color"])
        # Flip the window
        self.window.flip()
