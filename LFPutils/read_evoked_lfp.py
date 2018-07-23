"""
Uploaded to Github on Tuesday, Aug 1st, 2017

author: Tansel Baran Yasar

Contains the function for reading the stimulus-evoked LFP for a recording session.

Usage: through the main function in main.py script
"""

from utils.filtering import *
from utils.reading_utils import *
from utils.load_intan_rhd_format import *
from matplotlib.pyplot import *
from utils.OpenEphys import *
from tqdm import tqdm
import pickle
import h5py

def extract_stim_timestamps_der(stim, experiment):
    stim_diff = np.diff(stim)
    stim_timestamps = np.where(stim_diff > 0)[0]

    #Cutting the triggers that happen too close to the beginning or the end of the recording session
    stim_timestamps = stim_timestamps[(stim_timestamps > (experiment.cut_beginning*experiment.sample_rate))]
    stim_timestamps = stim_timestamps[(stim_timestamps < (len(stim) - experiment.cut_end*experiment.sample_rate))]

    return stim_timestamps

def read_evoked_lfp_from_stim_timestamps(filtered_data, stim_timestamps, experiment, mode):
    if mode == 'whisker':
        evoked_pre = experiment.whisker_evoked_pre
        evoked_post = experiment.whisker_evoked_post
    elif mode == 'light':
        evoked_pre = experiment.light_evoked_pre
        evoked_post = experiment.light_evoked_post

    evoked = np.zeros((len(stim_timestamps), len(filtered_data), int(experiment.sample_rate/30*(evoked_pre+evoked_post))))
        #Saving the evoked LFP waveforms in an array
    for i in (range(len(stim_timestamps))):
        evoked[i,:,:] = filtered_data[:,int(stim_timestamps[i]-evoked_pre*experiment.sample_rate/30):int(stim_timestamps[i]+evoked_post*experiment.sample_rate/30)]

    return evoked

def read_evoked_lfp(group, session ,data):
    """This function processes the data traces for the specified probe and shank in a recording session to obtain
	the mean evoked LFP activity. It saves the evoked activity and the average evoked activity in a Pickle file. It
	supports the data from 'file per channel' (dat) and 'file per recording' (rhd) options of Intan software and the
	data recorded by Open Ephys software (cont).

        Inputs:
		coords: List including the coordinates of the shank or tetrode (either [height, shank] for tetrode configuration
			or [probe, shank] for linear configuration
            	p: Dictionary containing parameters (see main file)
		data: The numpy array that contains the data from either tetrode or shank in cases of tetrode or linear configurations
			respectively

        Outputs:
            Saves the evoked LFP waveforms in a numpy array (number of trigger events x number of electrodes per shank x number
		of samples in the evoked LFP window) and the time stamps of the stimulus trigger events in a pickle file saved in the folder
		for the particular probe and shank of the analysis.
    """
    experiment = session.subExperiment.experiment
    probe = experiment.probe
    nr_of_electrodes = len(probe.id[group])
    f = h5py.File(experiment.dir + '/analysis_results.hdf5', 'a')
    f[session.subExperiment.name + '/group_{:g}/'.format(group) + session.name].create_dataset("downsampled_LFP", data = data)
    #Low pass filtering
    filt = lowpassFilter(rate = experiment.sample_rate/30, high = experiment.low_pass_freq, order = 4, axis = 1)
    filtered = filt(data)

    #Notch filtering
    if experiment.notch_filt_freq != 0:
        notchFilt = notchFilter(rate = experiment.sample_rate, low = experiment.notch_filt_freq-5, high = experiment.notch_filt_freq+5, order = 4, axis = 1)
        filtered = notchFilt(filtered)

    #Reading the trigger timestamps (process varies depending on the file formatd

    if experiment.fileformat == 'dat':
        if session.preferences['do_whisker_stim_evoked'] == 'y':
            whisker_trigger_filepath = session.whisker_stim_channel
            with open(whisker_trigger_filepath, 'rb') as fid:
                whisker_stim_trigger = np.fromfile(fid, np.int16)
                whisker_stim_timestamps = (extract_stim_timestamps_der(whisker_stim_trigger,experiment) / 30)
                whisker_stim_timestamps = whisker_stim_timestamps.astype('int')

                f[session.subExperiment.name + '/group_{:g}/'.format(group) + session.name].create_dataset("whisker_stim_timestamps", data = whisker_stim_timestamps)

        if session.preferences['do_optical_stim_evoked'] == 'y':
            optical_trigger_filepath = session.optical_stim_channel
            with open(optical_trigger_filepath, 'rb') as fid:
                optical_stim_trigger = np.fromfile(fid, np.int16)
                optical_stim_timestamps = (extract_stim_timestamps_der(optical_stim_trigger, experiment)/30)
                optical_stim_timestamps = optical_stim_timestamps.astype('int')

                f[session.subExperiment.name + '/group_{:g}/'.format(group) + session.name].create_dataset("optical_stim_timestamps", data  = optical_stim_timestamps)

    elif experiment.fileformat == 'cont':
        #Reading the digital input from file
        trigger_filepath = session.dir + '/all_channels.events'
        trigger_events = loadEvents(trigger_filepath)

        #Acquiring the timestamps of the ttl pulses
        timestamps = trigger_events['timestamps']
        eventId = trigger_events['eventId']
        eventType = trigger_events['eventType']
        channel = trigger_events['channel']

        timestamps_global = timestamps[eventType == 5]
        timestamps_ttl = []

        ttl_events = (eventType == 3)
        ttl_rise = (eventId == 1)

        for i in range(len(timestamps)):
            if (ttl_events[i]) and (ttl_rise[i]):
                timestamps_ttl = np.append(timestamps_ttl, timestamps[i])

        stim_timestamps = timestamps_ttl - timestamps_global[0]

    elif experiment.fileformat == 'rhd':
        trigger_all = []
        for file in range(session.rhd_files):
            data = read_data(session.dir+'/'+ session.rhd_files[file])
            trigger = data['board_dig_in_data'][1]
            trigger_all = np.append(trigger_all, trigger)

        stim_timestamps = []
        for i in range(1,len(trigger_all)):
            if trigger_all[i-1] == 0 and trigger_all[i] == 1:
                stim_timestamps = np.append(stim_timestamps, i)

    if session.preferences['do_whisker_stim_evoked'] == 'y':
        whisker_evoked = read_evoked_lfp_from_stim_timestamps(filtered, whisker_stim_timestamps, experiment, 'whisker')
        whisker_stim_grp = f[session.subExperiment.name + '/group_{:g}/'.format(group) + session.name].create_group("whisker_evoked_LFP")
        analyze_evoked_LFP(whisker_evoked, session, group, 'whisker', whisker_stim_grp)

    if session.preferences['do_optical_stim_evoked'] == 'y':
        optical_evoked = read_evoked_lfp_from_stim_timestamps(filtered, optical_stim_timestamps, experiment, 'light')
        optical_stim_grp = f[session.subExperiment.name + '/group_{:g}/'.format(group) + session.name].create_group("optical_evoked_LFP")
        analyze_evoked_LFP(optical_evoked, session, group, 'optical', optical_stim_grp)

def analyze_evoked_LFP(evoked, session, group, mode, grp):
    experiment = session.subExperiment.experiment

    if mode == 'whisker':
        evoked_pre = experiment.whisker_evoked_pre
        evoked_post = experiment.whisker_evoked_post

    elif mode == 'optical':
        evoked_pre = experiment.light_evoked_pre
        evoked_post = experiment.light_evoked_post

    time = np.linspace(-evoked_pre*1000, evoked_post*1000, (evoked_post + evoked_pre) * experiment.sample_rate/30)
    evoked_svg_path = session.subExperiment.dir + '/analysis_files/group_{:g}/'.format(group)

    evoked_avg = np.mean(evoked,0) #Average evoked LFP waveforms across trials
    evoked_std = np.std(evoked, 0) #Standard deviation of the evoked LFP waveforms across trials
    evoked_err = evoked_std / math.sqrt(len(evoked)) #Standard error of the evoked LFP waveforms across trials

    grp.create_dataset("mean_{:s}_evoked_LFP".format(mode), data = evoked_avg)
    grp.create_dataset("standard_deviation_{:s}_evoked_LFP".format(mode), data=evoked_std)
    grp.create_dataset("standard_error_{:s}_evoked_LFP".format(mode), data = evoked_err)
    grp.create_dataset("{:s}_evoked_LFP".format(mode), data = evoked)

    for trode in range(len(evoked[0])):
        trode_index = session.probe.id[group][trode]
        figure()
        plot(time, evoked_avg[trode], 'k-')
        fill_between(time, evoked_avg[trode]-evoked_err[trode], evoked_avg[trode]+evoked_err[trode])
        xlabel('Time (ms)')
        ylabel('Voltage (uV)')

        ylim_min = np.floor(np.min(evoked) / 100) * 100
        ylim_max = np.ceil(np.max(evoked) / 100) * 100
        ylim(ylim_min, ylim_max)
        savefig(evoked_svg_path + 'electrode{:g}_{:s}_evoked.svg'.format(trode_index,mode), format = 'svg')
        close()
