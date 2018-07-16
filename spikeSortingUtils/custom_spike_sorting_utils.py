import numpy as np
from glob import glob
from utils.filtering import *
from utils.load_intan_rhd_format import *
from utils.reading_utils import *

def read_location(location, group):

    #Initializing variables and inheriting objects
    experiment = location.experiment
    end_inds = np.zeros(len(location.sessions)+1)
    channels = experiment.probe.id[group]
    data = np.zeros((len(channels), 0))
    waveforms = np.zeros((0,len(channels),(experiment.spike_samples_before + experiment.spike_samples_after)))
    peak_times = np.zeros(0)
    time = np.zeros(0)
    chunk_size = experiment.chunk_size

    #.dat file that keeps the binary data for the location
    data_file = open(glob(location.dir, '/analysis_files/group_{:g}/group_{:g}.dat'.format(group, group)), 'rb')

    #Creating a merged time array for the location
    current_time = 0
    for session_index in (location.sessions):
        session = location.sessions[session_index]
        time_session = read_time_dat_file(glob(session.dir, '/time.dat'), experiment.sample_rate) + current_time
        time = np.append(time, time_session, 0)
        current_time = time_session[-1]
        end_inds[session_index+1] = end_inds[session_index] + len(time_session)
    location.time = time
    location.end_inds = end_inds

    #Reading the data from .dat file by chunks
    reading_index = 0
    while reading_index < len(time_session):
        chunk_time = time[reading_index:reading_index + chunk_size] #Time interval of the chunk in s
        chunk = np.fromfile(data_file, filetype = 'int16', len(channels)*chunk_size) #Data for the chunk in compressed format
        reshaped_chunk = np.reshape(chunk, (chunk_size, len(channels))) #Reshaping the chunk into a 2d array
        reshaped_chunk = np.transpose(reshaped_chunk) #Transposing the chunk to get the N_electrodes x N_samples format in the array
        filtered_chunk = experiment.bandfilt(reshaped_chunk) #Bandpass filtering the chunkchannels

        if reading_index == 0:
            location.noise = np.sqrt(np.mean(np.square(filterered_chunk)))

        (waveforms_chunk, peak_times_chunk) = extract_waveforms(filtered_chunk, chunk_time, location) #Extracting waveforms and peak times from the chunk
        waveforms = np.append(waveforms, waveforms_chunk, 0) #Appending the waveforms from the chunk to the waveforms for the location
        peak_times = np.append(peak_times, peak_times_chunk + current_end_time) #Appending the peak times from the chunk to the peak times for the location
        reading_index = reading_index + chunk_size #Iterating reading index by the amount of chunk size

        location_output = {'waveforms': waveforms, 'peak_times':peak_times, 'time':time, 'end_inds':end_inds}

    return location_output

def extract_waveforms(data, time, location):
	"""
	This function extracts waveforms from multiple channel data based on the
	given threshold coefficient. A relative threshold is calculated by
	multiplying the rms of the recording with the given threshold coefficient.
	For each time step the function scans through all the electrodes to find an
	event of threshold crossing. If so, the waveforms in all channels surrounding
	that time step are recorded.

	Inputs:
		data: Numpy array containing the bandpass filtered data in the form of
			(N_electrodes x N_time steps)
		params: Dictionary containing the recording parameters. The following
			entries must be present:

			sample_rate: Sampling rate in Hz
			time: Time array of the recording (numpy array)
			pre: Extent of the spike time window prior to peak (in ms)
			post: Extent of the spike time window post the peak (in ms)
			flat_map_array: Flattened numpy array containing the spatial
				information of the electrodes (contains only one element
				for single channel recordings)
			threshold_coeff: Coefficient used for calculating the threshold
				value per channel

	Outputs:
		waveforms: Numpy array containing the extracted waveforms from all
			channels (N_events x N_electrodes x N_timesteps_in_spike_time_range)
		peak_times: numpy array containing the times of the events (in s)
			(1xN_events)
	"""

    experiment = location.experiment
	threshold = noise * experiment.threshold_coeff

	waveforms = []
	peak_times = []
	found = False
	for i in range(experiment.spike_samples_before, len(data[0]) - experiment.spike_samples_after):
		for trode in range(len(data)):
			if (abs(data[trode,i) > threshold[trode]) and (abs(data[trode,i]) > abs(data[trode,i-1])) and (abs(data[trode,i]) > abs(data[trode,i+1])):
				found = True
				break
		waveform = data[:,(i - experiment.spike_samples_before):(i + experiment.spike_samples_after)]
		waveforms.append(waveform)
		peak_times.append(time[i])
		found = False

	waveforms = np.asarray(waveforms)
	peak_times = np.asarray(peak_times)
	return waveforms, peak_times
