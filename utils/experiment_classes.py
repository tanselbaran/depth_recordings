from glob import glob
import os
import numpy as np
from importlib import import_module

class Experiment:
    def __init__(self, experiment_dir, type):
        self.dir = experiment_dir
        self.name = self.dir.split('/')[-1]
        self.locations = {}
        self.type = type

    def add_subExperiment(self, subExperiment_dir):
        index_subExperiment = len(self.subExperiments)
        self.subExperiments[index_subExperiment] = subExperiment(subExperiment_dir, self)
        print(subExperiment_dir)
        self.subExperiment[index_subExperiment].amplifier_port = input('Please enter the amplifier port to which the amplifier is connected to')
        preferences = {}
        preferences['do_spike_analysis'] = self.get_input_for_pref("Do spike detection, sorting and post-processing for this session? (y/n)")
        self.subExperiments[index_subExperiment].preferences = preferences

    def createProbe(self, probe_file_dir, probe_name):
        cwd = os.getcwd()
        os.chdir(probe_file_dir)
        probe_module = import_module(probe_name)
        probe_class = getattr(probe_module, probe_name)
        os.chdir(cwd)

        self.probe = probe_class()
        self.probe.get_channel_mapping(self.amplifier)
        self.probe.get_channel_coords()

    def get_input_for_pref(self,statement):
        while True:
            inpt = input(statement)
            if (inpt == 'y' or inpt == 'n'):
                break
            else:
                print("Invalid input! Please enter a valid input (y or n).")
        return inpt


class Session:
    def __init__(self, session_name, subExperiment):
        self.name = session_name
        self.dir = subExperiment.dir + '/' + session_name
        self.subExperiment = subExperiment

    def get_input_for_pref(self,statement):
        while True:
            inpt = input(statement)
            if (inpt == 'y' or inpt == 'n'):
                break
            else:
                print("Invalid input! Please enter a valid input (y or n).")
        return inpt

    def set_analysis_preferences(self):
        preferences = {}
        print(self.name)
        preferences['do_whisker_stim_evoked'] = self.get_input_for_pref("Do whisker stimulation evoked analysis for this session? (y/n)")
        ref_channels = self.get_input_for_pref("Which channels will be used for software referencing to detect spikes?")
        ref_channels = np.asarray(ref_channels.split(','))
        preferences['ref_channels'] = ref_channels.astype('int8')
        self.preferences = preferences

    def set_amplifier(self):
        info_file = glob(self.dir + '/info*')[0]
        info_file_suffix = info_file.split('.')[-1]
        self.amplifier = info_file_suffix

    def setTrigChannels(self, *args):
        self.set_amplifier()
        if self.amplifier == 'rhd':
            prefix = 'board-DIN-0'
        elif self.amplifier == 'rhs':
            prefix = 'board-DIGITAL-IN-0'
        if self.preferences['do_whisker_stim_evoked'] == 'y':
            self.whisker_stim_channel = self.dir + '/' + prefix + str(args[0]) + '.dat'

    def createProbe(self, amplifier, probe_file_dir, probe_name):
        cwd = os.getcwd()
        os.chdir(probe_file_dir)
        probe_module = import_module(probe_name)
        probe_class = getattr(probe_module, probe_name)
        os.chdir(cwd)

        self.probe = probe_class()
        self.probe.get_channel_mapping(amplifier)
        self.probe.get_channel_coords()

class subExperiment:
    def __init__(self, subExperiment_dir, experiment):
        self.dir = subExperiment_dir
        self.name = self.dir.split('/')[-1]
        self.sessions = {}
        self.experiment = experiment

    def add_session(self, session_name, order):
        index_session = len(self.sessions)
        self.sessions[index_session] = Session(session_name, self)
        self.sessions[index_session].set_analysis_preferences()
        self.sessions[index_session].order = order

    def add_sessions_in_dir(self):
        subdirs = sorted(glob(self.dir + "/*[!analysis]"))
        for subdir in subdirs:
            current_session = 1
            session_name = subdir.split('/')[-1]
            if len(subdirs) == 1:
                order = 3
            elif current_session == 1:
                order = 0
            elif current_session == len(subdirs):
                order = 2
            else:
                order = 1
            self.add_session(session_name, order)
            current_session = current_session + 1

class Probe:
    def __init__(self, probe_name):
        self.name = probe_name
        self.probe_module = importlib.import_module('probe_files.'+probe_name)
