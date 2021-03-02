import os
from importlib import import_module
from utils.reading_utils import *
from glob import glob

class Experiment:
    #
    def __init__(self, experiment_dir):
        """
        Initialization function of the Experiment class. This class contains the parameters and classes for the overall acute or chronic experiment.

        Inputs:
            -experiment_dir: The directory to the folder that contains the recording session folders for the experiment
        """
        self.dir = experiment_dir
        self.name = self.dir.split('/')[-1]
        self.sessions = {}
        #self.type = type

    def createProbe(self, probe_file_dir, probe_name):
        """
        This function is for getting the probe information (type, mapping etc.) from the pre-defined probe file .py. The probe information is acquired from this file by importing the probe object that is defined in that file.

        Inputs:
        -probe_file_dir: Directory of the .py file containing the information on the probe used in the experiment.
        -probe_name: Name of the probe used in the experiment.

        Outputs:
        -self.id: Mapping of the channel indices from the amplifier mapping to the inherent organization in the probe.
        -self.coords: Array containing the coordinates of the channels (N arrays of two elements for N channels, where each array is [x,y] coordinates of the channel)
        """
        cwd = os.getcwd() #Store the current working directory
        os.chdir(probe_file_dir) #Go to the working directory of the probe file
        probe_module = import_module(probe_name) #Import the probe object from the probe file
        probe_class = getattr(probe_module, probe_name) #Import the probe class
        os.chdir(cwd)

        self.probe = probe_class()
        self.probe.get_channel_mapping(self.amplifier) #Get the channel mapping from the probe file and save it in the probe object of the Experiment
        self.probe.get_channel_coords() #Get the coordinates of the channels

    """def get_input_for_pref(self,statement):
        while True:
            inpt = input(statement)
            if (inpt == 'y' or inpt == 'n'):
                break
            else:
                print("Invalid input! Please enter a valid input (y or n).")
        return inpt"""

    def add_session(self, session_name):
        """
        This function creates a Session object for a session name specified under an experiment.

        Inputs:
        -session_name (string): Name of the session that is to be created.

        """
        index_session = len(self.sessions)
        self.sessions[index_session] = Session(session_name, self) #Initialize the Session object and add it to the sessions dictionary of the experiment.
        #self.sessions[index_session].amplifier_port = input('Please enter the amplifier port to which the amplifier is connected to') #Get user input on which amplifier port was used for this session.
        self.sessions[index_session].set_preprocessing_preferences()

    def add_sessions_in_dir(self):
        """
        This function detects all the sessions listed in the experiment directory and creates a Session object for each one of them by using the add_session function.
        """
        subdirs = [d for d in os.listdir(self.dir) if (os.path.isdir(os.path.join(self.dir, d)) and d != '__pycache__')]
        subdirs = sorted(subdirs)
        for subdir in subdirs:
            session_name = subdir.split('/')[-1]
            self.add_session(session_name)

class Session:
    #This class contains the recording session that was performed as part of an Experiment.
    def __init__(self, session_name, experiment):
        """
        Initialization function for the session object. Inputs needed for initialization:
        -session_name: string, name of the recording session
        -experiment: Experiment object, which the Session is associated with.
        """
        self.name = session_name
        self.dir = experiment.dir + '/' + session_name
        self.experiment = experiment

    def get_input_for_pref(self,statement):
        while True:
            inpt = input(statement)
            if (inpt == 'y' or inpt == 'n'):
                break
            else:
                print("Invalid input! Please enter a valid input (y or n).")
        return inpt

    def set_preprocessing_preferences(self):
        """
        This function gets the preprocessing preferences for spike sorting and sensory-evoked responses.
        """
        """cwd = os.getcwd() #Store the current working directory
        os.chdir(self.experiment.dir) #Go to the working directory of the probe file
        preprocessing_pref_module = import_module('preprocessing_preferences') #Import the preprocessing_prefereces module from the preprocessing_preferences.py file
        self.preferences = preprocessing_pref_module.preferences[self.name]
        #self.preferences = getattr(preprocessing_pref_module, preferences) #Import the preprocessing_preferences dictionary
        os.chdir(cwd) #go back to the original working directory"""

        preferences = {}
        print(self.name)
        preferences['do_spike_analysis'] = self.get_input_for_pref("Do spike detection, sorting and post-processing for this session? (y/n)")
        preferences['do_whisker_stim_evoked'] = self.get_input_for_pref("Do whisker stimulation evoked analysis for this session? (y/n)")
        ref_channels = input("Which channels will be used for software referencing to detect spikes?")
        self.ref_channels = ref_channels.split(',')
        dead_channels = input("Which channels are dead?")
        self.dead_channels = dead_channels.split(',')
        if self.ref_channels != ['']:
            self.ref_channels = np.asarray(self.ref_channels)
            self.ref_channels = self.ref_channels.astype('int8')
        if self.dead_channels != ['']:
            self.dead_channels = np.asarray(self.dead_channels)
            self.dead_channels = self.dead_channels.astype('int8')
        self.preferences = preferences

    def set_amplifier(self):
        """
        This function obtains the information on which amplifier was used, (rhd vs. rhs) and sets it as a parameter for the session accordingly.
        """
        info_file = glob(self.dir + '/info*')[0]
        info_file_suffix = info_file.split('.')[-1]
        self.amplifier = info_file_suffix

    def setTrigChannels(self, *args):
        """
        This function stores the digital input channel number used for whisker stimulation in the parameter of whisker_stim_channel, if the whisker stimulation related analysis is enabled for a particular session.
        """
        self.set_amplifier()
        if self.amplifier == 'rhd':
            prefix = 'board-DIN-0'
        elif self.amplifier == 'rhs':
            prefix = 'board-DIGITAL-IN-0'
        if self.preferences['do_whisker_stim_evoked']:
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

    def get_duration(self):
        """This function gets the duration of the recording session in seconds, by using the time.dat file and the sampling rate, and saves it as the duration variable within the Session object.
        """
        time_file_dir = self.dir + '/time.dat'
        session_time = read_time_dat_file(time_file_dir, self.sample_rate)
        self.duration = session_time[-1]

    #def get_downsampled_LFP(self):
        #generate the empty array
        #read one channel, decimate and write into the empty array
        #write the array into a .dat file


class Probe:
    def __init__(self, probe_name):
        self.name = probe_name
        self.probe_module = importlib.import_module('probe_files.'+probe_name)
