# -*- coding: utf-8 -*-
# ======================================================================================== #
#
#
#                    SCRIPT: log_handler.py
#
#
#          DESCRIPTION: Custom Logger
#
#
#                       RULE: DAYW
#
#
#
#                  CREATOR: Sharif Saleki
#                         TIME: 04-23-2024-7810598105114117
#                       SPACE: Dartmouth College, Hanover, NH
#
# ======================================================================================== #
import logging
import inspect
import traceback


class DaedalusLogger(logging.Logger):
    """
    A class to create and configure a logger for Daedalus package.
    """
    def __init__(self, name, enable_debug=False, log_file=None):
        """
        Initialize the logger.

        Args:
            name (str): The name of the logger.
            debug (bool): Whether to run in debug mode.
            log_file (str): The path to the log file.
        """
        super().__init__(name)
        self.enable_debug = enable_debug

        # Add handlers
        if log_file is not None:
            self.add_file_handler(log_file)
        self.add_console_handler()

    def debug(self, msg, *args, **kwargs):
        kwargs.setdefault('stacklevel', 3)
        super().debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        kwargs.setdefault('stacklevel', 3)
        super().info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        kwargs.setdefault('stacklevel', 3)
        super().warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        kwargs.setdefault('stacklevel', 3)
        super().error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        kwargs.setdefault('stacklevel', 3)
        super().critical(msg, *args, **kwargs)

    def findCaller(self, stack_info=True, stacklevel=2):
        """
        Find the stack frame of the caller so that we can note the source file name,
        line number, and function name. Override to account for custom wrapper methods.
        """
        f = inspect.currentframe()
        if f is not None:
            f = f.f_back  # skip the frame of this method itself

        while f and stacklevel > 0:
            if f.f_code.co_name == "log_trial":  # Specific method to skip
                stacklevel += 1  # Skip additional frame if inside a known wrapper
            f = f.f_back
            stacklevel -= 1
        if f is not None:
            co = f.f_code
            sinfo = None
            if stack_info:
                sinfo = traceback.format_stack(f)
            return co.co_filename, f.f_lineno, co.co_name, sinfo
        return "(unknown file)", 0, "(unknown function)", None

    def add_file_handler(self, log_file):
        """
        Adds handlers to the logger.

        Args:
            log_file (str): The path to the log file.
        """
        # Create handler
        hand = logging.FileHandler(log_file)

        # Set level
        level = logging.DEBUG
        hand.setLevel(level)

        # Create formatters and add it to handlers
        fmter = CustomFormatter("file", level)
        hand.setFormatter(fmter)

        # Add handlers to the logger
        self.addHandler(hand)

    def add_console_handler(self):
        """
        Adds a console handler to the logger.
        """
        # Create handler
        hand = logging.StreamHandler()

        # Set level
        # level = logging.DEBUG if self.enable_debug else logging.WARNING
        level = logging.DEBUG
        hand.setLevel(level)

        # Create formatters and add it to handlers
        fmter = CustomFormatter("conosle", level)
        hand.setFormatter(fmter)

        # Add handlers to the logger
        self.addHandler(hand)

    def show_handler_info(self):
        """
        Show the information of the handlers.
        """
        for handler in self.handlers:
            print(f"Handler: {handler}")
            print(f"Level: {logging.getLevelName(handler.level)}")
            print(f"Formatter: {handler.formatter}")

    def set_handlers_level(self, handler_type, new_level):
        """
        Set the level of the handlers.

        Args:
            handler_type (str): The type of the handler.
            new_level (str): The new level for the handler.

        Raises:
            ValueError: If the handler type is not found.
        """
        level = self.get_log_level(new_level)
        for handler in self.handlers:
            if (handler_type == "file") and (isinstance(handler, logging.FileHandler)):
                handler.setLevel(level)
                fmter = CustomFormatter(logger_device=handler_type, logger_level=level)
                handler.setFormatter(fmter)
            elif (handler_type == "console") and (isinstance(handler, logging.StreamHandler)):
                handler.setLevel(level)
                fmter = CustomFormatter(logger_device=handler_type, logger_level=level)
                handler.setFormatter(fmter)
            else:
                raise ValueError(f"Handler {handler_type} not found")

    def get_handlers_level(self, handler_type):
        """
        Get the level of the handlers.

        Args:
            handler_type (str): The type of the handler.

        Returns:
            str: The level of the handler.
        """
        for handler in self.handlers:
            if (handler_type == "file") and (isinstance(handler, logging.FileHandler)):
                return logging.getLevelName(handler.level)
            elif (handler_type == "console") and (isinstance(handler, logging.StreamHandler)):
                return logging.getLevelName(handler.level)

    def get_log_level(self, level):
        """
        Get the logging level from a string.

        Args:
            level (str): The log level as a string.

        Returns:
            int: The corresponding logging level.
        """
        levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        return levels.get(level.upper(), logging.DEBUG)

    def remove_all_handlers(self):
        for handler in list(self.handlers):
            self.removeHandler(handler)

    def close_file(self):
        for handler in self.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()


# Register the custom logger
logging.setLoggerClass(DaedalusLogger)


class CustomFormatter(logging.Formatter):
    """
    Define a custom log formatter
    """
    def __init__(self, logger_device, logger_level, fmt=None):
        if fmt is None:
            fmt = "%(asctime)s | %(levelname)s | %(message)s"
        super().__init__(fmt)
        self.logger_device = logger_device
        self.logger_level = logger_level

    def format(self, record):
        """
        Define fixed widths for different parts of the log message

        Args:
            record (logging.LogRecord): The log record to format

        Returns:
            str: The formatted log message
        """
        # Format the log level and function name with fixed widths
        func_width = 20
        level_width = 8
        # Ensure level name is capped at the fixed width
        # level = f"{record.levelname}".ljust(level_width)[:level_width]
        level = f"{record.levelname}".ljust(level_width)[:level_width]
        # Ensure function name is capped at the fixed width
        # function = f"{record.funcName}".ljust(func_width)[:func_width]
        function = f"{record.funcName}:{record.lineno}".ljust(func_width)
        # Format time consistently
        time = self.formatTime(record, "%Y-%m-%d %H:%M:%S")

        # Create the formatted log message
        log_msg = ""
        if self.logger_device == "file":
            log_msg += f"{time} | "
        if self.logger_level == logging.DEBUG:
            log_msg += f"@{function} | "
        log_msg += f"{level} | "
        if hasattr(record, "block"):
            log_msg += f"BLOCK {record.block} | "
        if hasattr(record, "trial"):
            log_msg += f"TRIAL {record.trial} | "
        log_msg += f"{record.msg}"

        return log_msg
