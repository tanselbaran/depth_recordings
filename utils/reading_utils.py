"""
Created on Tuesday, Aug 1st 2017

author: Tansel Baran Yasar

Contains the utilities for reading the data from different file formats created by Intan and Open Ephys
GUI softwares.
"""

import numpy as np
from utils.load_intan_rhd_format import *
import os
import h5py

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

def extract_stim_timestamps_der(stim, session):
    """
    This function extracts the timestamps in index values from the digital input data of the Intan board.

    Inputs:
        stim: Array that contains the signal from the digital input channel for the stimulus ttl pulse
        session: The session object for which the stimulus timestamps are extracted

    Outputs:
        stim_timestamps: Stimulus timestamps (defined by the rising edge of the TTL pulse) in the units of samples (divided by the predefined downsampling factor)
    """

    stim_diff = np.diff(stim)
    stim_timestamps = np.where(stim_diff > 0)[0]
    experiment = session.experiment

    #Cutting the triggers that happen too close to the beginning or the end of the recording session
    stim_timestamps = stim_timestamps[(stim_timestamps > (experiment.cut_beginning*session.sample_rate))]
    stim_timestamps = stim_timestamps[(stim_timestamps < (len(stim) - experiment.cut_end*session.sample_rate))]

    return stim_timestamps

def read_stimulus_trigger(session):
    """
    This function extracts the stimulus timestamps from the TTL signal file for a session and saves it as an array in the preprocessing hdf file.

    Inputs:
        session: The session for which the stimulus information is to be retrieved.

    Outputs:
        stimulus timestamps are saved in the hdf file with the unit of samples divided by the predefined downsampling factor for the LFP analysis
    """
    experiment = session.experiment
    f = h5py.File(experiment.dir + '/preprocessing_results.hdf5', 'a')

    if experiment.fileformat == 'dat':
        whisker_trigger_filepath = session.whisker_stim_channel
        #Read data from the .dat file containing the TTL signal
        with open(whisker_trigger_filepath, 'rb') as fid:
            whisker_stim_trigger = np.fromfile(fid, np.int16)

        #Detect the timestamps from the TTL signal
        whisker_stim_timestamps = (extract_stim_timestamps_der(whisker_stim_trigger,session) / session.downsampling_factor)
        whisker_stim_timestamps = whisker_stim_timestamps.astype('int')

        #Save the stim timestamps in the hdf file
        f[session.name].create_dataset("whisker_stim_timestamps", data = whisker_stim_timestamps)

def read_channel(session, group, trode, chunk_inds):
    """
    TO DO
    -Write documentation
    -Prevent reading the whole data file every single time a chunk is going to be read
    """
    experiment = session.experiment

    if experiment.fileformat == 'dat':
        #For the "channel per file" option of Intan
        if trode < 10:
            prefix = '00'
        else:
            prefix = '0'
        electrode_path = session.dir + '/amp-' + session.amplifier_port + '-' +prefix + str(int(trode)) + '.dat'
        electrode_data = read_amplifier_dat_file(electrode_path)

        if chunk_inds[1] == -1:
            electrode_data = electrode_data[chunk_inds[0]:]
        else:
            electrode_data = electrode_data[chunk_inds[0]:chunk_inds[1]]

    return electrode_data

"""def return_ch_idx(ch_number):
    if ch_number < 10:
        ch_idx = '00{:g}'.format(ch_number)
    else:
        ch_idx = '0{:g}'.format(ch_number)
    return ch_idx"""

"""def save_downsampled_data(session, group, preprocessing_file, spike_sorting_preprocessing_files_dir):
    experiment = session.experiment
    probe_id = deepcopy(experiment.probe.id)

    channels = probe_id[group]
    if session.preferences['dead_channels'] != ['']:
        for dead_channel in sorted(session.preferences['dead_channels'], reverse=True):
            channels.remove(dead_channel)

    time = read_time_dat_file(session.dir + '/time.dat', session.sample_rate)

    data_all_downsampled = np.memmap(spike_sorting_preprocessing_files_dir + '/group_{:g}/group_{:g}_temp_filtered.dat'.format(group,group), dtype='int16', mode='w+', shape=(len(channels),int(len(time)/session.downsampling_factor)))

    for i, ch in enumerate(channels):
        ch_idx = return_ch_idx(ch)
        raw_data = read_amplifier_dat_file(session.dir + '/amp-A-{:}.dat'.format(ch_idx))
        raw_data_ds = signal.decimate(raw_data, session.downsampling_factor)
        data_all_downsampled[i] = raw_data_ds

    ch_grp = preprocessing_file[session.name+'/group_{:g}'.format(group)]
    ch_grp.create_dataset("downsampled_LFP", data=data_all_downsampled)

    os.remove(spike_sorting_preprocessing_files_dir + '/group_{:g}/group_{:g}_temp_filtered.dat'.format(group,group))"""


def read_group_into_dat_file(session, group, spike_sorting_preprocessing_files_dir):
    #Writing the data into the .dat file if spike sorting will be performed.
    time = read_time_dat_file(session.dir + '/time.dat', session.sample_rate)
    if (session.ref_channels != ['']):
        reference = np.zeros((len(session.ref_channels), len(time)))
        for i, ref_channel in enumerate(session.ref_channels):
            reference[i] = read_channel(session, group, ref_channel, [0,-1])
        mean_reference = np.mean(reference, 0)
    else:
        mean_reference = np.zeros(len(time))

    data_all = np.memmap(spike_sorting_preprocessing_files_dir + '/group_{:g}/group_{:g}_temp.dat'.format(group,group),dtype='int16', mode='w+', shape=(len(session.good_channels),len(time)))

    if np.array_equal(session.good_channels, session.ref_channels):
        data_all = reference - mean_reference

    else:
        for i, ch in enumerate(session.good_channels):
            raw_data = read_channel(session, group, ch, [0,-1])
            refd_data = raw_data - mean_reference
            data_all[i] = refd_data

    data_all = data_all.flatten('F')
    data_all.tofile(open(spike_sorting_preprocessing_files_dir + '/group_{:g}/group_{:g}.dat'.format(group, group), 'wb'))

    os.remove(spike_sorting_preprocessing_files_dir + '/group_{:g}/group_{:g}_temp.dat'.format(group,group))
