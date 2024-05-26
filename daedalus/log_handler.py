#!/usr/bin/env python
"""
created 4/23/24

@author Sharif Saleki

Logging configuration for the LSS analysis.
"""
import logging


def get_logger(name):
    """
    Make a logger.

    Returns:

    """
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    return logger


def add_handlers(logger, log_file, *extra_handlers):
    """

    Returns:
        logger (logging.Logger): The logger object.
    """
    # Create handlers (console and file handler)
    c_handler = logging.StreamHandler()
    f_handlers = [logging.FileHandler(log_file)]
    if extra_handlers:
        f_handlers.append(logging.FileHandler(handler) for handler in extra_handlers)

    # Set levels for handlers
    c_handler.setLevel(logging.WARNING)
    f_handlers[0].setLevel(logging.INFO)
    for handler in f_handlers[1:]:
        handler.setLevel(logging.DEBUG)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    for handler in f_handlers:
        handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    for handler in f_handlers:
        logger.addHandler(handler)

    return logger


def set_handlers_level(logger, handler_type, new_level):
    for handler in logger.handlers:
        if isinstance(handler, handler_type):
            handler.setLevel(new_level)


def remove_handlers(logger):
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

