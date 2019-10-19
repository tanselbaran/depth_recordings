"""
Created on Tuesday, Aug 1st 2017

author: Tansel Baran Yasar

Contains the utilities for reading the data from different file formats created by Intan and Open Ephys
GUI softwares.
"""

import numpy as np
from utils.load_intan_rhd_format import *
import os
import pickle
from scipy import signal
import h5py
from copy import deepcopy

def read_amplifier_dat_file(filepath):
    """
    This function reads the data from a .dat file created by Intan software and returns as a numpy array.

    Inputs:
        filepath: The path to the .dat file to be read.

    Outputs:
        amplifier_file: numpy array containing the data from the .dat file (in uV)
    """

    with open(filepath, 'rb') as fid:
        raw_array = np.fromfile(fid, np.int16)
    amplifier_file = raw_array * 0.195 #converting from int16 to microvolts
    return amplifier_file

def read_time_dat_file(filepath, sample_rate):
    """
    This function reads the time array from a time.dat file created by Intan software for a recording session,
    and returns the time as a numpy array.

    Inputs:
        filepath: The path to the time.dat file to be read.
        sample_rate: Sampling rate for the recording of interest.

    Outputs:
        time_file: Numpy array containing the time array (in s)
    """

    with open(filepath, 'rb') as fid:
        raw_array = np.fromfile(fid, np.int32)
    time_file = raw_array / float(sample_rate) #converting from int32 to seconds
    return time_file

def extract_stim_timestamps_der(stim, experiment):
    stim_diff = np.diff(stim)
    stim_timestamps = np.where(stim_diff > 0)[0]

    #Cutting the triggers that happen too close to the beginning or the end of the recording session
    stim_timestamps = stim_timestamps[(stim_timestamps > (experiment.cut_beginning*experiment.sample_rate))]
    stim_timestamps = stim_timestamps[(stim_timestamps < (len(stim) - experiment.cut_end*experiment.sample_rate))]

    return stim_timestamps

def read_stimulus_trigger(session):
    experiment = session.subExperiment.experiment
    stim_timestamps = {}
    f = h5py.File(experiment.dir + '/analysis_results.hdf5', 'a')

    if experiment.fileformat == 'dat':

        if session.preferences['do_whisker_stim_evoked'] == 'y':
            whisker_trigger_filepath = session.whisker_stim_channel
            with open(whisker_trigger_filepath, 'rb') as fid:
                whisker_stim_trigger = np.fromfile(fid, np.int16)
            whisker_stim_timestamps = (extract_stim_timestamps_der(whisker_stim_trigger,experiment) / experiment.downsampling_factor)
            whisker_stim_timestamps = whisker_stim_timestamps.astype('int')
            stim_timestamps['whisker_stim_timestamps'] = whisker_stim_timestamps
            f[session.subExperiment.name + '/' + session.name].create_dataset("whisker_stim_timestamps", data = whisker_stim_timestamps)

    return stim_timestamps

def read_channel(session, group, trode, chunk_inds):
    """
    TO DO
    -Write documentation
    -Prevent reading the whole data file every single time a chunk is going to be read
    """
    experiment = session.subExperiment.experiment
    probe = experiment.probe
    id = probe.id #Reading the channel id file from the parameters dictionary
    electrode_index = id[group][trode]

    if experiment.fileformat == 'dat':
        #For the "channel per file" option of Intan
        if electrode_index < 10:
            prefix = '00'
        else:
            prefix = '0'
        electrode_path = session.dir + '/amp-' + session.subExperiment.amplifier_port + '-' +prefix + str(int(electrode_index)) + '.dat'
        electrode_data = read_amplifier_dat_file(electrode_path)

        if chunk_inds[1] == -1:
            electrode_data = electrode_data[chunk_inds[0]:]
        else:
            electrode_data = electrode_data[chunk_inds[0]:chunk_inds[1]]

    return electrode_data

def return_ch_idx(ch_number):
    if ch_number < 10:
        ch_idx = '00{:g}'.format(ch_number)
    else:
        ch_idx = '0{:g}'.format(ch_number)
    return ch_idx

def read_group_into_dat_file(session, group, spike_sorting_analysis_files_dir):
    #Writing the data into the .dat file if spike sorting will be performed.
    #if session.subExperiment.preferences['do_spike_analysis'] == 'y':
    experiment = session.subExperiment.experiment
    probe_id = deepcopy(experiment.probe.id)

    channels = probe_id[group]
    if session.dead_channels != ['']:
        for dead_channel in sorted(session.dead_channels, reverse=True):
            channels.remove(dead_channel)

    time = read_time_dat_file(session.dir + '/time.dat', experiment.sample_rate)
    if session.ref_channels != ['']:
        reference = np.zeros((len(session.ref_channels), len(time)))
        for i, ref_channel in enumerate(session.ref_channels):
            ch_idx = return_ch_idx(ref_channel)
            reference[i] = read_amplifier_dat_file(session.dir + '/amp-A-{:}.dat'.format(ch_idx))
        mean_reference = np.mean(reference, 0)
    else:
        mean_reference = np.zeros(len(time))

    data_all = np.memmap(spike_sorting_analysis_files_dir + '/group_{:g}/group_{:g}_temp.dat'.format(group,group),dtype='int16', mode='w+', shape=(len(channels),len(time)))
    for i, ch in enumerate(channels):
        ch_idx = return_ch_idx(ch)
        raw_data = read_amplifier_dat_file(session.dir + '/amp-A-{:}.dat'.format(ch_idx))
        refd_data = raw_data - mean_reference
        data_all[i] = refd_data
    data_all = data_all.flatten('F')
    data_all.tofile(open(spike_sorting_analysis_files_dir + '/group_{:g}/group_{:g}.dat'.format(group, group), 'wb'))
    os.remove(spike_sorting_analysis_files_dir + '/group_{:g}/group_{:g}_temp.dat'.format(group,group))
