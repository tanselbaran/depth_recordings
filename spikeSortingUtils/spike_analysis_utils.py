"""
Created on Friday, Oct 6th, 2017

author: Tansel Baran Yasar

Contains the functions for analyzing spike trains.
"""
import numpy as np
from matplotlib.pyplot import *
from LFPutils.read_evoked_lfp import *
import h5py

def generate_psths(location, group, spike_trains_location):
    experiment = location.experiment
    f = h5py.File(experiment.dir + '/analysis_results.hdf5', 'a')
    ses_grp = f[location.name + '/group_{:g}/'.format(group) + session.name]
    for session_index in location.sessions:
        analysis_prefs = location.sessions[session_index].preferences
        if analysis_prefs['do_whisker_stim_evoked'] == 'y':
            whisker_stim_timestamps = ses_grp["whisker_stim_timestamps"]
            subsession_end_inds =session.break_down_to_subsessions(whisker_stim_timestamps)
            whisker_bounds = [experiment.whisker_evoked_pre, experiment.whisker_evoked_post]
            whisker_evoked_trains, whisker_evoked_psths = get_psth(spike_trains_location[session_index], session, whisker_stim_timestamps, whisker_bounds, subsession_end_inds, session.fake_whisker_stim)
            ses_grp.create_dataset("whisker_evoked_trains", data = whisker_evoked_trains)
            ses_grp.create_dataset("whisker_evoked_psths", data = whisker_evoked_psths)

        if analysis_prefs['do_optical_stim_evoked'] == 'y':
            optical_stim_timestamps = ses_grp["optical_stim_timestamps"]
            subsession_end_inds = session.break_down_to_subsessions(optical_stim_timestamps)
            optical_bounds = [experiment.light_evoked_pre, experiment.light_evoked_post]
            light_evoked_trains, light_evoked_psths = get_psth(spike_trains_location[session_index], session, optical_stim_timestamps, optical_bounds, subsession_end_inds, session.fake_optical_stim)
            ses_grp.create_dataset("optical_evoked_trains", data = optical_evoked_trains)
            ses_grp.create_dataset("optical_evoked_psths", data = whisker_evoked_psths)

def break_down_to_sessions(location, spike_times, spike_trains):
    end_inds = location.end_inds
    spike_trains_location = {}
    for session_index in location.sessions:
        session = location.sessions[session_index]
        spike_trains_session = spike_trains[end_inds[session_index]:end_inds[session_index+1]]
        spike_trains_session = np.astype(spike_trains_session, 'int8')
        spike_trains_location[session_index] = spike_trains_session

    return spike_trains_location

def firing_histogram(bin_size, spike_times, sample_rate, end_inds):
    """
    This function generates a histogram of the given spike trains with the given bin sizes.

    Inputs:
        bin_size: The bin size for the histogram (in s)
        spike_times: Dictionary containing the time indicies of the spikes to be analyzed
        sample_rate: Sampling rate of the data (in Hz)
        end_inds:  The indices of the stim-spont or anesthesia level transitions in the recording trace (first entry and last entry must be 0 and the length of recording trace in samples respectively)

    Outputs:
        hist: Array containing the histogram of the spiking activity (Nx[total number of bins in the histogram])
    """
    bin_size_inds = sample_rate * bin_size #converting bin size from seconds to samples
    hist = np.zeros((len(spike_times), int(end_inds[-1]/bin_size_inds)))
    spike_train = np.zeros(int(end_inds[-1]))
    for unit in range(len(hist)):
        spike_train[spike_times[unit]] = 1
        for bin in range(len(hist[0])):
            hist[unit, bin] = np.sum(spike_train[int(bin*bin_size_inds):int(min(end_inds[-1]*sample_rate, (bin+1)*bin_size_inds))]) #the end boundary of the interval is set by checking if the next bin would exceed the last index of the recoring trace.

    return hist

def get_firing_rate(spike_times, end_inds, sample_rate):
    """
    This function calculates the average firing rates for the units for which the spike trains are given, for a given interval.

    Inputs:
        spike_times: Dictionary containing the time indicies of the spikes to be analyzed
        sample_rate: Sampling rate of the data (in Hz)
        end_inds: The indices of the stim-spont or anesthesia level transitions in the recording trace

    Outputs:
        firing_rate: Array containing the firing rates for the units at each epoch (NxM array where N is the number of units and M is the number of epochs in the recording.
    """
    spike_train = np.zeros(int(end_inds[-1]))
    firing_rate = np.zeros((len(spike_times), len(end_inds)-1))

    for unit in range(len(spike_times)):
        spike_train[spike_times[unit]] = 1
        for epoch in range(len(end_inds)-1):
            rec_time_len = (end_inds[epoch+1] - end_inds[epoch]) / sample_rate #calculating the length of the epoch
            firing_rate[unit][epoch] = np.sum(spike_train[int(end_inds[epoch]):int(end_inds[epoch+1])]) / rec_time_len #calculating the average firing rate of the unit in the epoch
        spike_train = np.zeros(int(end_inds[-1])) #Resetting the spike train before analyzing the next unit

    return firing_rate


def get_psth(spike_trains, session, stim_timestamps, bounds, subsession_end_inds, fake_stim):
    """
    This function generates the PSTHs of the spike trains of the units provided for the given epochs of stimulation and anesthesia levels.

    Inputs:
        spike_trains: Spike train(s) to be analyzed. (Nx[total number of data samples] where N is the number of units for which the spike trains are analyzed)
        stim: Array containing the digital input trace for the stimulus
        bounds: The milliseconds to be included in the PSTH prior and post the stimulus trigger
        end_inds: The indices of the stim-spont or anesthesia level transitions in the recording trace
        bin_size: The bin size for the PSTH (in s)
        sample_rate: Sampling rate of the data (in Hz)

    Outputs:
        PSTH: The PSTHs of the units for the given epochs in an array of size KxLxMxN  where K is the number of units, L is the number of epochs, M is the number of stimulus trigger events for that epoch and N is the number of samples in the range for one PSTH
    """

    experiment = session.subExperiment.experiment
    sample_rate = experiment.sample_rate
    bin_size = experiment.bin_size

    psth_range = np.arange(-bounds[0],bounds[1],bin_size) #range for the interval of psth bins
    evoked_range = np.arange(-bounds[0],bounds[1],1/sample_rate) #range for the evoked spike trains
    bin_size_inds = int(sample_rate * bin_size) #converting bin size from seconds to samples
    num_subsessions = len(subsession_end_inds) + 1

    stim_timestamps = np.append(stim_timestamps, fake_stim[0])
    stim_timestamps = np.append(stim_timestamps, fake_stim[1])

    for unit in range(len(spike_trains)):
        evoked_trains = {}
        evoked_psths = {}
        for subsession in range(num_subsessions):
            stim_timestamps_subsession = stim_timestamps[np.where(np.logical_and((stim_timestamps > subsession_end_inds[subsession]), (stim_timestamps < subsession_end_inds[subsession])))] #Extracting the stim indices that fall inside the epoch of interest
            evoked_train = np.zeros((len(stim_timestamps_subsession), len(evoked_range)))
            evoked_psth = np.zeros((len(subsession_end_inds), len(psth_range)))
            for i, stim_ind in enumerate(stim_inds_epoch):
                evoked_train[i] = spike_train[(stim_ind-bounds[0]*sample_rate/1000.):(stim_ind+bounds[1]*sample_rate/1000.)]
            for bin in range(len(psth_range)):
                evoked_psth[bin] = np.sum(evoked_train[:,bin*bin_size_inds:(bin+1)*bin_size_inds])
            evoked_trains[subsession] = evoked_train
            evoked_psths[subsession] = evoked_psth
    return evoked_trains, evoked_psths


def plot_firing_histogram(hist, unit, bin_size, end_inds, sample_rate):
    """
    This function plots the firing rate for a given unit.

    Inputs:
        hist: Array containing the histogram of the spiking activity (Nx[total number of bins in the histogram]) generated by the firing_histogram function
        unit: The index of unit for which the firing rate would be plotted
        bin_size: The bin size for the PSTH (in s)
        end_inds: The indices of the stim-spont or anesthesia level transitions in the recording trace
    """

    figure()
    plot(np.arange(0,len(hist[unit])*bin_size, bin_size), hist[unit]/bin_size)
    for i in range(len(end_inds)-1):
        axvline(end_inds[i+1]/sample_rate, color = 'r', linestyle = 'dashed')
    xlabel('Time (s)')
    ylabel('Firing rate (Hz)')
    show()

def plot_psth(unit, evoked_psth, psth_range):
    figure()f[location.name + '/group_{:g}/'.format(group) + session.name+"/optical_stim_timestamps"]*
    plot(psth_range, np.mean(evoked_psth, 0))
    axvline(0, color = 'r', linestyle = 'dashed')
    xlabel('Time (ms)')
    ylabel('Voltage (uV)')
    show()

def plot_spike_train(spike_trains, time, stim):
    fig, axs = subplots(2,1, sharex = 'all', figsize = (10,10))
    axs[0].plot(time, spike_trains[unit])
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Spike train')
    axs[1].plot(time, stim)
    show()
