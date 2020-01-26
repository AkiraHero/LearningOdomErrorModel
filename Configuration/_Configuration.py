#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/08
# @Author  : Akira Ju
# @Email   : jukaka@pku.edu.cn
# @File    : _Configuration.py

import argparse
import os
import sys
from collections import OrderedDict as odict
from configparser import ConfigParser, ExtendedInterpolation

from ._Setting import Setting


class Configuration:
    """
    Class Configuration: base class for a set of configuration using by machine learning.

    """
    _config = ConfigParser(dict_type=odict, interpolation=ExtendedInterpolation())
    reinit = False
    loaded_flag = None

    def __init__(self, filepath=None, reinit=False):
        """
        Initialize the Configuration class.
        :param filepath: configuration file. Reconstruct a new empty file if filepath=None.
        :param reinit: if true, force to make a new empty file.
        """
        self._keys = []
        self.__init_setting_section__()
        self.reinit = reinit
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        if filepath is None:
            self.parser.add_argument('default_configfile', type=str, help="The default configuration file.")
            filepath = self.parser.parse_known_args()[0].default_configfile
        if os.path.exists(filepath):
            if not self.reinit:
                if not self.load_config(filepath):
                    raise ValueError("Cannot load configuration file. Format Error!")
                self.ini_options()
                self.rewrite_options()
            else:
                pathname = os.path.dirname(sys.argv[0])
                if filepath[0] is not '/':
                    filepath = os.path.join(pathname, filepath)
                self.gen_empty_configure(filepath)
                self.reinit = True
        else:
            pathname = os.path.dirname(sys.argv[0])
            if filepath[0] is not '/':
                filepath = os.path.join(pathname, filepath)
            self.gen_empty_configure(filepath)
            self.reinit = True

    def __setattr__(self, key, value):
        """
        store new attribute (key, value) pairs in builtin __dict__
        :param key:
        :param value:
        """
        self.__dict__[key] = value
        # store the keys in self._keys in the order that they are initialized
        # do not store '_keys' itelf and don't enter any key more than once
        if key not in ['_keys'] + self._keys:
            self._keys.append(key)

    @classmethod
    def __init_setting_section__(cls):
        """
        Virtual function for customization
        """
        raise NotImplementedError

    def _get_sections(self):
        """
        Internal function for collecting section(built-in class instance).
        :return: List of section variables.
        """
        return [i for i in self._keys if isinstance(self.__dict__[i], Setting)]

    def _load_sections(self, section):
        """
        Internal function for section loading.
        :param section:
        """
        assert isinstance(section, Setting)
        section.__init__(config=self._config, name=section.get_section_name())

    def load_config(self, filepath):
        """
        Load configuration from file.
        :param filepath: string, filename
        :return: Bool, success flag
        """
        self._config.read(filepath)
        section_list = self._get_sections()
        self.loaded_flag = True

        for section in section_list:
            section_instance = self.__dict__[section]
            self._load_sections(section_instance)
            self.loaded_flag = self.loaded_flag and section_instance.is_loaded()
        return self.loaded_flag

    def gen_empty_configure(self, filepath):
        """
        Generate an empty configuration template.
        :param filepath: string, filename
        """
        section_list = self._get_sections()
        for section in section_list:
            name = self.__dict__[section].get_section_name()
            self._config[name] = {}
            for i in self.__dict__[section].get_variables():
                self._config[name][i] = ''

        with open(filepath, 'w') as configfile:
            self._config.write(configfile)
        print('Generate Empty Configuration file:', filepath)

    def write_config(self, filepath):
        """
        Write configuration to file.
        :param filepath: string, filename
        """
        with open(filepath, 'w') as configfile:
            self._config.write(configfile)

    def display_all(self):
        """
        Display the content of configuration.
        """
        section_list = self._get_sections()
        for section in section_list:
            name = self.__dict__[section].get_section_name()
            print('-------' + name + '-------')
            for i in self.__dict__[section].get_variables():
                print('\t', i + '=', self.__dict__[section].get_attribute_value(i))

    def display_key_setting(self):
        """
        Virtual function for customization
        """
        raise NotImplementedError

    def str2bool(self, v):
        """
        Add new definition for Bool variable recognition of argparse
        :param v: string, option name
        :return:
        """
        if v.lower() in ('yes', 'true', '1'):
            return True
        elif v.lower() in ('no', 'false', '0'):
            return False
        else:
            raise NotImplementedError

    def ini_options(self):
        """
        Initialize argparser according to configuration file
        """
        # initialize option list
        section_list = self._get_sections()
        for section in section_list:
            name = self.__dict__[section].get_section_name()
            arg_group = self.parser.add_argument_group(title=name)
            for i in self.__dict__[section].get_variables():
                default_value = self.__dict__[section].get_attribute_value(i)
                if section == 'validating_dataset':
                    keyword = '--v' + i
                else:
                    keyword = '--' + i
                if default_value.__class__ is bool:
                    arg_group.add_argument(keyword, type=self.str2bool, default=default_value)
                else:
                    arg_group.add_argument(keyword, type=default_value.__class__, default=default_value)

    def rewrite_options(self):
        """
        Rewrite the configuration variable inside this class according to the input from command line
        """
        options, unknown = self.parser.parse_known_args()
        if len(unknown):
            raise NameError("Unknown options:" + str(unknown))
        section_list = self._get_sections()
        for section in section_list:
            for i in self.__dict__[section].get_variables():
                if section == 'validating_dataset':
                    if 'v' + i in self.parser.parse_known_args()[0]:
                        self.__dict__[section].set_attribute_value(i, options.__getattribute__('v' + i))
                else:
                    if i in self.parser.parse_known_args()[0]:
                        self.__dict__[section].set_attribute_value(i, options.__getattribute__(i))

        # rewrite self._config
        for section in section_list:
            name = self.__dict__[section].get_section_name()
            for i in self.__dict__[section].get_variables():
                self._config.set(section=name, option=i, value=str(self.__dict__[section].get_attribute_value(i)))
