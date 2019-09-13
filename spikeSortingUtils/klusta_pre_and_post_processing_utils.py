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

### Klustakwik utilities for data analyzing ###

def create_prm_file(group,subExperiment):
    """
    This function creates the .prm file required by Klustakwik for the analysis of the data from the tetrode or shank of a linear probe with the group index s.

    Inputs:
        probe: Index of the probe, for recordings with multiple probes.
        s: Group index; shanks in the case of linear probes (0 for the left-most shank, looking at the electrode side), tetrodes in the case of tetrode organization (starting from the left and bottom-most tetrode first and increasing upwards column by column)
        p: Parameters dictionary containing the parameters and preferences related to spike sorting.
    """
    experiment = subExperiment.experiment
    probe = experiment.probe
    file_dir = subExperiment.dir + '/analysis_files/group_{:g}/group_{:g}.prm'.format(group,group)

    with open(file_dir, 'a') as text:
        print('experiment_name = \group_{:g}\''.format(group), file = text)
        print('prb_file = \'group_{:g}.prb\''.format(group), file = text)
        print('traces = dict(raw_data_files=[experiment_name + \'.dat\'],voltage_gain=10., sample_rate =' + str(experiment.sample_rate) + ', n_channels = ' + str(probe.nr_of_electrodes_per_group) + ', dtype = \'int16\')', file = text)

        print("spikedetekt = { \n 'filter_low' : %d., \n 'filter_high_factor': 0.95 * .5, \n 'filter_butter_order': {:}, \n #Data chunks. \n 'chunk_size_seconds': 1., \n 'chunk_overlap_seconds': .015, \n #Threshold \n 'n_excerpts': 50, \n 'excerpt_size_seconds': 1., \n 'use_single_threshold': True, \n 'threshold_strong_std_factor': {:}, \n 'threshold_weak_std_factor': 2., \n 'detect_spikes': 'negative', \n # Connected components. \n 'connected_component_join_size': 1, \n #Spike extractions. \n 'extract_s_before': {:}, \n 'extract_s_after': {:}, \n 'weight_power': 2, \n #Features. \n 'n_features_per_channel': 3, \n 'pca_n_waveforms_max': 10000}".format(experiment.bandfilter_order, experiment.threshold_coeff, experiment.spike_samples_before, experiment.spike_samples_after) , file = text)

        print("klustakwik2 = { \n 'prior_point':1, \n 'mua_point':2, \n 'noise_point':1, \n 'points_for_cluster_mask':100, \n 'penalty_k':0.0, \n 'penalty_k_log_n':1.0, \n 'max_iterations':1000, \n 'num_starting_clusters':500, \n 'use_noise_cluster':True, \n 'use_mua_cluster':True, \n 'num_changed_threshold':0.05, \n 'full_step_every':1, \n 'split_first':20, \n 'split_every':40, \n 'max_possible_clusters':1000, \n 'dist_thresh':4, \n 'max_quick_step_candidates':100000000, \n 'max_quick_step_candidates_fraction':0.4, \n 'always_split_bimodal':False, \n 'subset_break_fraction':0.01, \n 'break_fraction':0.0, \n 'fast_split':False, \n 'consider_cluster_deletion':True, \n #'num_cpus':None, \n #'max_split_iterations':None \n}", file = text)

    text.close()

#For post-processing of the spikes and clusters determined by Klustakwik
def retain_cluster_info(probe,group,p):
    """
    This function extracts the spike info from the clu file output of Klustakwik that was run on a tetrode data and saves the spike times and waveforms for  a cluster in a pickle file.

    Inputs:
        group: index of channel groups (0 for the left-most and bottom-most, increasing upwards first in a shank and rightwards when the shank is complete)
        p: parameters dictionary for the recording session

    Outputs:
        Saves a pickle file in the same folder as the kwik file. The dictionary contains the following;
            P: spike info dictionary which consist of C sub-dictionaries where C is the number of clusters. Each sub-dictionary consists of the following:
                spike_times_cluster: times of the spikes that belong to this cluster
                waveforms: waveforms of the spikes that belong to this cluster
            p: Params dictionary from the rest of the pipeline
    """

    path_kwik_file = p['mainpath'] + '/analysis_files/probe_{:g}_group_{:g}/probe_{:g}_group_{:g}.kwik'.format(probe,group,probe,group) #path to the kwik file
    with h5py.File(path_kwik_file,'r') as hf:
        all_spiketimes = hf.get('channel_groups/0/spikes/time_samples') #accessing the spike times
        np_all_spiketimes = np.array(all_spiketimes) #converting the spike times to numpy array

    path_clu_file = p['mainpath'] + '/analysis_files/probe_{:g}_group_{:g}/probe_{:g}_group_{:g}.clu.0'.format(probe,group,probe,group) #accessing the file where the clustering information is stored
    np_clu_original = np.loadtxt(path_clu_file) #reading out the clustering information, which is an array with (number of spikes + 1) entries...
    nr_of_clusters = int(np_clu_original[0]) #... whose first entry is the number of clusters. As a result, we need to...
    np_clu = np_clu_original[1:] # ...separate the actual clustering information by excluding the first element.

    raw_data = np.fromfile(p['mainpath'] + '/analysis_files/probe_{:g}_group_{:g}/probe_{:g}_group_{:g}.dat'.format(probe,group,probe,group),dtype='int16') #reading out the raw data file
    num_samples = int(len(raw_data) / p['nr_of_electrodes_per_group'])
    raw_data = np.reshape(raw_data, (num_samples, p['nr_of_electrodes_per_group']))
    fil = bandpassFilter(rate=p['sample_rate'], low=p['low_cutoff'], high=p['high_cutoff'], order=3, axis = 0)
    raw_data_f = fil(raw_data)
    units = {}

    unit_indices = np.unique(np_clu)
    unit_indices = np.delete(unit_indices, [0,1])
    for cluster in unit_indices:
        spike_times_cluster_index = np.where(np_clu == cluster)
        spike_times_cluster = np_all_spiketimes[spike_times_cluster_index]
        num_spikes_in_cluster = len(spike_times_cluster)
        num_samples_per_waveform = p['samples_before'] + p['samples_after']
        waveforms = np.zeros((num_spikes_in_cluster,p['nr_of_electrodes_per_group'],num_samples_per_waveform))
        for spike in range(num_spikes_in_cluster):
            for trode in range(p['nr_of_electrodes_per_group']):
                for sample in range(num_samples_per_waveform):
                    waveforms[spike,trode,sample] = raw_data_f[(int(spike_times_cluster[spike])-p['samples_before']+sample), trode]

        unit = [0,0]
        unit[0] = spike_times_cluster
        unit[1] = waveforms
        units['unit{:g}'.format(cluster)] = unit

    path_pickle_file = p['mainpath'] + '/analysis_files/probe_{:g}_group_{:g}/probe_{:g}_group_{:g}_spikeinfo.pickle'.format(probe,group,probe,group)
    with open (path_pickle_file, 'wb') as f:
        pickle.dump({'units':units, 'params_dict':p} ,f)
