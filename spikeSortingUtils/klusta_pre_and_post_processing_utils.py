"""
Creation date: Tuesday, Aug 1st 2017

authors: Tansel Baran Yasar and Clemens Dlaska

Contains the functions that are used for creating the .prm and .prb files required for spike-sorting with Klustakwik, and running the Klustakwik on the tetrode or shank (for linear probes).
"""

import numpy as np
import os as os
import h5py
import json
import pickle
from scipy import signal
from utils.filtering import *
import itertools

### Klustakwik utilities for data analyzing ###

def create_prm_file(group,session):
    """
    This function creates the .prm file required by Klustakwik for the analysis of the data from the tetrode or shank of a linear probe with the group index s.

    Inputs:
        probe: Index of the probe, for recordings with multiple probes.
        s: Group index; shanks in the case of linear probes (0 for the left-most shank, looking at the electrode side), tetrodes in the case of tetrode organization (starting from the left and bottom-most tetrode first and increasing upwards column by column)
        p: Parameters dictionary containing the parameters and preferences related to spike sorting.
    """
    experiment = session.subExperiment.experiment
    probe = experiment.probe
    file_dir = experiment.dir + '/analysis_files/' + session.subExperiment.name + '/' + session.name + '/spike_sorting/group_{:g}/group_{:g}.prm'.format(group,group)

    with open(file_dir, 'a') as text:
        print('experiment_name = \'group_{:g}\''.format(group), file = text)
        print('prb_file = \'group_{:g}.prb\''.format(group), file = text)
        print('traces = dict(raw_data_files=[\'group_{:g}.dat\'], sample_rate ='.format(group) + str(experiment.sample_rate) + ', n_channels = ' + str(probe.nr_of_electrodes_per_group) + ', dtype = \'int16\')', file = text)

        print("spikedetekt = { \n 'filter_low' : %d., \n 'filter_high_factor': 0.95 * .5, \n 'filter_butter_order': %i, \n #Data chunks. \n 'chunk_size_seconds': 1., \n 'chunk_overlap_seconds': .015, \n 'n_excerpts': 50, \n 'excerpt_size_seconds': 1., \n 'use_single_threshold': True, \n 'threshold_strong_std_factor': %d, \n 'threshold_weak_std_factor': 2., \n 'detect_spikes': 'negative', \n #Connected components. \n 'connected_component_join_size': 1, \n #Spike extractions. \n 'extract_s_before': %i, \n 'extract_s_after': %i, \n 'weight_power': 2, \n #Features. \n 'n_features_per_channel': 3, \n 'pca_n_waveforms_max':10000}" % (experiment.low_cutoff_bandpass, experiment.bandfilter_order, experiment.threshold_coeff, experiment.spike_samples_before, experiment.spike_samples_after) , file = text)

        print("klustakwik2 = dict( \n prior_point=1, \n mua_point=2, \n noise_point=1, \n  points_for_cluster_mask=100, \n penalty_k=0.0, \n penalty_k_log_n=1.0, \n max_iterations=1000, \n num_starting_clusters=500, \n use_noise_cluster=True, \n use_mua_cluster=True, \n num_changed_threshold=0.05, \n full_step_every=1, \n split_first=20, \n split_every=40, \n max_possible_clusters=1000, \n dist_thresh=4, \n max_quick_step_candidates=100000000, \n max_quick_step_candidates_fraction=0.4, \n  always_split_bimodal=False, \n subset_break_fraction=0.01, \n break_fraction=0.0, \n fast_split=False, \n consider_cluster_deletion=True, \n num_cpus=6)", file = text)

    text.close()

def create_linear_prb_file(group, session, neighboorhood=3):
    experiment = session.subExperiment.experiment
    probe = experiment.probe
    file_dir = experiment.dir + '/analysis_files/' + session.subExperiment.name + '/' + session.name + '/spike_sorting/group_{:g}/group_{:g}.prb'.format(group,group)

    channels = list(range(probe.nr_of_electrodes_per_group))
    for channel in session.dead_channels:
        channels.remove(channel)

    adjacency = []
    for i in channels:
        if (i + neighborhood) < probe.nr_of_electrodes_per_group:
            for j in range(neighborhood):
                if (i in channels) and (i+j+1 in channels):
                    adjacency.append((i,i+j+1))
        else:
            for j in range(nr_of_electrodes_per_group - i):
                if (i in channels) and (i+j+1 in channels):
                    adjacency.append((i,i+j+1))

    geometry = {}
    for i in channels:
        geometry[i] = (0, i*10)

    with open(file_dir, 'a') as text:
        print('channel_groups = {', file = text)
        print("0: {:}'channels':".format('{') + str(channels), file=text)
        print("'graph':"+str(adjacency), file = text)
        print("'geometry: '"+str(geometry)+"}}", file = text)

    text.close()

#For post-processing of the spikes and clusters determined by Klustakwik
def get_spike_times_and_waveforms_from_unit(unit_id, major_electrode, data_filtered, sr, spike_time_pre, spike_time_post, cluster_id, spike_idx, align=True):
    unit_spikes = np.where(cluster_id == unit_id)[0]
    unit_spike_times = spike_idx[unit_spikes]
    waveforms = np.zeros((len(unit_spikes), len(data_filtered), int(2*(spike_time_pre+spike_time_post)*sr/1000)))
    for i, spike_time in enumerate(unit_spike_times):
        waveforms[i,:,:] = data_filtered[:,int(spike_time-(2*(spike_time_pre)*sr/1000)):int(spike_time+(2*spike_time_post*sr/1000))]

    if align:
        waveforms, index_correction = align_waveforms_at_peak(major_electrode, waveforms, sr, spike_time_pre, spike_time_post)
        unit_spike_times = unit_spike_times + index_correction

    return waveforms, unit_spike_times

def align_waveforms_at_peak(major_electrode, waveforms, sr, spike_time_pre, spike_time_post):
    peak_indices = np.zeros(len(waveforms))
    for i in range(len(waveforms)):
        peak_indices[i] = np.where(np.abs(waveforms[i,major_electrode,int(spike_time_pre*sr/1000):int((2*spike_time_pre+spike_time_post)*sr/1000)]) == np.max(np.abs(waveforms[i,major_electrode,int(spike_time_pre*sr/1000):int((2*spike_time_pre+spike_time_post)*sr/1000)])))[0][0] + int(spike_time_pre*sr/1000)

    aligned_waveforms = np.zeros((len(waveforms), len(waveforms[0]), int((spike_time_pre+spike_time_post)*sr/1000)))
    for i,spike_time in enumerate(peak_indices):
        aligned_waveforms[i,:,:] = waveforms[i,:,int(spike_time-(spike_time_pre*sr/1000)):int(spike_time+(spike_time_post*sr/1000))]

    index_correction = peak_indices - 2*spike_time_pre*sr/1000
    return aligned_waveforms, index_correction

def get_all_spike_info(units, session, analysis_file_dir):
    experiment = session.subExperiment.experiment
    spike_sorting_folder = experiment.dir + '/analysis_files/' + session.subExperiment.name + '/' + session.name + '/spike_sorting/group_0/'
    kwik_file = h5py.File(spike_sorting_folder + 'group_0.kwik', 'r')
    cluster_id = np.asarray(kwik_file['channel_groups/0/spikes/clusters/main'])
    spike_idx = np.asarray(kwik_file['channel_groups/0/spikes/time_samples'])

    data = np.fromfile(spike_sorting_folder + '/group_0.dat', dtype = 'int16')
    data = np.reshape(data, (experiment.probe.nr_of_electrodes, int(len(data)/experiment.probe.nr_of_electrodes)), 'F')

    b,a = signal.iirfilter(4, [600/experiment.sample_rate, 6000/experiment.sample_rate], btype='bandpass')
    data_filtered = signal.filtfilt(b,a, data, axis=1)

    analysis_file = h5py.File(analysis_file_dir, 'r+')
    for unit in range(len(units)):
        unit_grp = analysis_file[session.subExperiment.name + '/' + session.name + '/group_0/'].create_group(str(units[unit][0]))
        waveforms, spike_times = get_spike_times_and_waveforms_from_unit(units[unit][0],units[unit][1],data_filtered,experiment.sample_rate,(1000*experiment.spike_samples_before/experiment.sample_rate),(1000*experiment.spike_samples_after/experiment.sample_rate), cluster_id, spike_idx)
        unit_grp.create_dataset("waveforms", data=waveforms)
        unit_grp.create_dataset("spike_times", data=spike_times)
