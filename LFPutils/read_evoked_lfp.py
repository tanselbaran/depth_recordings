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
from tqdm import tqdm
import pickle
import h5py

def read_evoked_lfp_from_stim_timestamps(filtered_data, stim_timestamps, experiment, mode):

    ds_sample_rate = experiment.sample_rate/experiment.downsampling_factor
    evoked_pre = int(getattr(experiment, "{:s}_evoked_pre".format(mode))*ds_sample_rate)
    evoked_post = int(getattr(experiment, "{:s}_evoked_post".format(mode))*ds_sample_rate)

    evoked = np.zeros((len(stim_timestamps), evoked_pre+evoked_post))
        #Saving the evoked LFP waveforms in an array
    for i in (range(len(stim_timestamps))):
        evoked[i,:] = filtered_data[int(stim_timestamps[i]-evoked_pre):int(stim_timestamps[i]+evoked_post)]

    return evoked

def read_evoked_lfp(group, session):
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
    ds_sample_rate = experiment.sample_rate/experiment.downsampling_factor

    if session.preferences['do_whisker_stim_evoked'] == 'y':
        whisker_stim = True
    else:
        whisker_stim = False

    if whisker_stim:
        whisker_stim_timestamps = f[session.subExperiment.name + '/' + session.name +  '/whisker_stim_timestamps']
        whisker_stim_grp = f[session.subExperiment.name + '/' + session.name + '/group_{:g}'.format(group)].create_group("whisker_evoked_LFP")
        whisker_evoked = np.zeros((nr_of_electrodes, len(whisker_stim_timestamps),  int(ds_sample_rate*(experiment.whisker_evoked_pre+experiment.whisker_evoked_post))))

    for trode in range(nr_of_electrodes):
        electrode_data = read_channel(session, group, trode, [0,-1])
        electrode_data = signal.decimate(electrode_data, experiment.downsampling_factor)
        f[session.subExperiment.name + '/' + session.name].create_dataset("channel_{:g}_downsampled_LFP".format(probe.id[group][trode]), data = electrode_data)

        #Low pass filtering
        filt = lowpassFilter(rate = ds_sample_rate, high = experiment.low_pass_freq, order = experiment.low_pass_order, axis = 0)
        filtered = filt(electrode_data)

        #Notch filtering
        if experiment.notch_filt_freq != 0:
            notchFilt = notchFilter(rate = ds_sample_rate, low = experiment.notch_filt_freq-5, high = experiment.notch_filt_freq+5, order = 4, axis = 0)
            filtered = notchFilt(filtered)

        if whisker_stim:
            whisker_evoked[trode] = read_evoked_lfp_from_stim_timestamps(filtered, whisker_stim_timestamps, experiment, 'whisker')

    if whisker_stim:
        analyze_evoked_LFP(whisker_evoked, session, group, 'whisker', whisker_stim_grp)

def analyze_evoked_LFP(evoked, session, group, mode, grp):
    experiment = session.subExperiment.experiment
    ds_sample_rate = experiment.sample_rate/experiment.downsampling_factor
    evoked_pre = getattr(experiment, "{:s}_evoked_pre".format(mode))
    evoked_post = getattr(experiment, "{:s}_evoked_post".format(mode))

    time = np.linspace(-evoked_pre*1000, evoked_post*1000, (evoked_post + evoked_pre) * ds_sample_rate)
    if not os.path.exists(session.subExperiment.dir + '/analysis_files/group_{:g}/'.format(group) + session.name):
        os.mkdir(session.subExperiment.dir + '/analysis_files/group_{:g}/'.format(group) + session.name)
    if experiment.probe.nr_of_electrodes_per_group == 1:
        if not os.path.exists(session.subExperiment.dir + '/analysis_files/' + session.name):
            os.mkdir(session.subExperiment.dir + '/analysis_files/' + session.name)
        evoked_svg_path = session.subExperiment.dir + '/analysis_files/' + session.name
    else:
        if not os.path.exists(session.subExperiment.dir + '/analysis_files/group_{:g}/'.format(group) + session.name):
            os.mkdir(session.subExperiment.dir + '/analysis_files/group_{:g}/'.format(group) + session.name)
        evoked_svg_path = session.subExperiment.dir + '/analysis_files/group_{:g}/'.format(group) + session.name
        
    evoked_avg = np.mean(evoked,1) #Average evoked LFP waveforms across trials
    evoked_std = np.std(evoked, 1) #Standard deviation of the evoked LFP waveforms across trials
    evoked_err = evoked_std / math.sqrt(len(evoked)) #Standard error of the evoked LFP waveforms across trials

    grp.create_dataset("mean_{:s}_evoked_LFP".format(mode), data = evoked_avg)
    grp.create_dataset("standard_deviation_{:s}_evoked_LFP".format(mode), data=evoked_std)
    grp.create_dataset("standard_error_{:s}_evoked_LFP".format(mode), data = evoked_err)
    grp.create_dataset("{:s}_evoked_LFP".format(mode), data = evoked)

    for trode in range(len(evoked)):
        trode_index = session.probe.id[group][trode]
        figure()
        plot(time, evoked_avg[trode], 'k-')
        fill_between(time, evoked_avg[trode]-evoked_err[trode], evoked_avg[trode]+evoked_err[trode])
        xlabel('Time (ms)')
        ylabel('Voltage (uV)')

        savefig(evoked_svg_path + '/electrode{:g}_{:s}_evoked.svg'.format(trode_index,mode), format = 'svg')
        close()
