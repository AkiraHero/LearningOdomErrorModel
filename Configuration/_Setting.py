#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/08
# @Author  : Akira Ju
# @Email   : jukaka@pku.edu.cn
# @File    : _Setting.py

import traceback
from functools import wraps


def check_none_wrap(func):
    """
    Check whether a function return None and raise ValueError.
    :param func:
    :return:
    """
    @wraps(func)
    def newfunc(*args, **kwargs):
        res = func(*args, **kwargs)
        if res is None:
            raise ValueError("None return. Empty value for keywords:{}".format(args[0]))
        return res
    return newfunc


class Setting:
    """
    Class Setting: base class for group of setting options.
    """
    def __init__(self, name, config=None):
        """
        Initialization.
        :param name: setting group name.
        :param config: Configuration instance.
        """
        self._keys = []
        self._section_name = name
        self._content = None
        self._loaded = False
        if config:
            self._content = config[self._section_name]
            # rewrite the function to check None return
            self._content.get = check_none_wrap(self._content.get)
            self._content.getint = check_none_wrap(self._content.getint)
            self._content.getboolean = check_none_wrap(self._content.getboolean)
            self._content.getfloat = check_none_wrap(self._content.getfloat)
            self._try_load(self._content)
        else:
            self._initialize_attrib()

    def __setattr__(self, key, value):
        # store new attribute (key, value) pairs in builtin __dict__
        self.__dict__[key] = value
        # store the keys in self._keys in the order that they are initialized
        # do not store '_keys' itelf and don't enter any key more than once
        if key not in ['_keys'] + self._keys:
            self._keys.append(key)

    def _try_load(self, _content):
        """
        Try to load setting.
        :param _content: content of Configuration.
        """
        try:
            self._load(_content)
            self._loaded = True
        except ValueError:
            self._loaded = False
            traceback.print_exc()
            print(self._section_name, ": fail to load.")

    def get_variables(self):
        return [i for i in self._keys if i[0] is not '_']

    def is_loaded(self):
        return self._loaded

    def get_section_name(self):
        return self._section_name

    def get_attribute_value(self, name):
        return self.__getattribute__(name)

    def set_attribute_value(self, key, value):
        self.__setattr__(key, value)

    @classmethod
    def _load(cls):
        raise NotImplementedError

    def _initialize_attrib(self):
        raise NotImplementedError
