#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =================================================================================================== #
#
#
#                        SCRIPT: managers.py
#
#
#               DESCRIPTION: Manage tasks and stuff
#
#
#                           RULE: DAYW
#
#
#
#                      CREATOR: Sharif Saleki
#                            TIME: 07-16-2024-7810598105114117
#                          SPACE: Dartmouth College, Hanover, NH
#
# =================================================================================================== #
from pathlib import Path
import shutil

from daedalus import utils


class BaseManager:
    """
    BaseManager class to handle base operations for the vision module
    """
    def __init__(self, **kwargs):

        self._all = []
        for key, val in kwargs.items():
            setattr(self, key.lower(), val)
            self._all.append(key)

    def add(self, **kwargs):
        for key, val in kwargs.items():
            key = key.lower()
            setattr(self, key, val)
            self._all.append(key)

    def get(self, name):
        """
        Get the object for a given name

        Args:
            name (str): The name of the object to get

        Returns:
            object: The object
        """
        name = name.lower()
        for attr in dir(self):
            if name in attr:
                return getattr(self, name)

    def show(self):
        for att in self._all:
            print(att)


class FileManager(BaseManager):
    """
    FileManager class to handle file operations
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _make_backup(self, file):
        """
        Handle file exists error

        Args:
            file_path (str): The file path that already exists
        """
        backup = file.with_suffix(".BAK")
        if backup.exists():
            backup.unlink()
        file.rename(backup)

    def iter_files(self):
        for file in self._all:
            yield file, getattr(self, file)

    def add(self, **kwargs):
        for key, val in kwargs.items():
            key = key.lower()
            val = Path(val)
            setattr(self, key, val)
            self._all.append(key)

    def make(self, **kwargs):
        for key, val in kwargs.items():
            key = key.lower()
            val = Path(val)
            if val.exists():
                self._make_backup(val)
            val.touch()
            setattr(self, key, val)


class DirectoryManager(BaseManager):
    """
    DirectoryManager class to handle directory operations for the vision module
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def add(self, **kwargs):
        for key, val in kwargs.items():
            key = key.lower()
            val = Path(val)
            setattr(self, key, val)
            self._all.append(key)

    def make(self, **kwargs):
        for key, val in kwargs.items():
            key = key.lower()
            val = Path(val)
            if not val.exists():
                val.mkdir(parents=True, exist_ok=True)
            setattr(self, key, val)
            self._all.append(key)

    def empty(self, *args):
        for name in args:
            name = name.lower()
            path_ = getattr(self, name)
            path_ = Path(path_)
            if not path_.is_dir():
                raise ValueError(f"The provided path {path_.name} doesn't exist.")

            for item in path_.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()


class SettingsManager(BaseManager):
    """
    Class to handle settings and parameters

    Args:
        platform (str): Platform for the module
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_from_file(self, name, file_path):
        """
        Load a configuration file

        Args:
            name (str): The name of the configuration file
            file_path (str): The path to the configuration file
        """
        name = name.lower()
        conf = utils.read_config(file_path)
        setattr(self, name, conf)
        self._all.append(name)


class DataManager(BaseManager):
    """
    DataManager class to handle data operations
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class SessionManager(BaseManager):
    """
    SessionManager class to handle session operations
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def add(self, **kwargs):
        for key, val in kwargs.items():
            key = key.lower()
            self._all.append(key)
            setattr(self, key, val)
            if key == "id":
                self.name = self.get_name(val)
                self._all.append("name")

    def get_name(self, ses_id):
        """
        Get the name of a session

        Args:
            ses_id (int): The session id

        Returns:
            str: The name of the session
        """
        return f"ses-{int(ses_id):02d}"


class SubjectManager(BaseManager):
    """
    SubjectManager class to handle subject operations
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def add(self, **kwargs):
        for key, val in kwargs.items():
            key = key.lower()
            setattr(self, key, val)
            self._all.append(key)
            if key == "id":
                self.name = self.get_name(val)
                self._all.append("name")

    def get_name(self, sub_id):
        """
        Get the name of a subject

        Args:
            sub_id (int): The subject id

        Returns:
            str: The name of the subject
        """
        return f"sub-{int(sub_id):02d}"
