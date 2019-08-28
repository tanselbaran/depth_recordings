import numpy as np
from glob import glob
from utils.filtering import *
from utils.load_intan_rhd_format import *
from utils.reading_utils import *
from scipy import signal
from tqdm import tqdm

def read_location(location, group):

    #Initializing variables and inheriting objects
    experiment = location.experiment
    end_inds = np.zeros((len(location.sessions),2))
    channels = experiment.probe.id[group]
    waveforms = np.zeros((0,len(channels),(experiment.spike_samples_before + experiment.spike_samples_after)))
    peak_times = np.zeros(0)
    time = np.zeros(0)
    data_all = np.zeros((len(channels), 0))
    chunk_size = int(experiment.chunk_size *  experiment.sample_rate)

    #Creating a merged time array for the location
    current_time = 0
    for session_index in (location.sessions):
        session = location.sessions[session_index]
        print(session.name)
        time_session = read_time_dat_file((session.dir+'/time.dat'), experiment.sample_rate)
        time = np.append(time, time_session+current_time, 0)
        if session_index == 0:
            end_inds[session_index] = [0,len(time_session)-1]
        else:
            end_inds[session_index] = [end_inds[session_index-1][1]+1, end_inds[session_index-1][1]+len(time_session)]
        noise = np.zeros(len(channels))
        reading_index = 0
        while reading_index < len(time_session):
            print('Reading chunk: {:g} / {:g}'.format(reading_index/chunk_size , int(len(time_session)/chunk_size)))
            if reading_index + chunk_size < len(time_session):
                data = np.zeros((len(channels), chunk_size))
                chunk_time = time_session[reading_index:(reading_index+chunk_size)]
                chunk_inds = [reading_index,reading_index+chunk_size]

            else:
                remaining_chunk_size = len(time_session) - reading_index
                data = np.zeros((len(channels), remaining_chunk_size))
                chunk_time = time_session[reading_index:]
                chunk_inds = [reading_index,-1]


            for trode in range(len(channels)):
                data[trode] = read_channel(session, group, trode,chunk_inds)
            filtered = experiment.bandfilt(data) #Bandpass filtering the data
            if reading_index == 0:
                noise = np.sqrt(np.mean(np.square(filtered), 1))
            (waveforms_chunk, peak_times_chunk) = extract_waveforms(filtered, chunk_time, location, noise) #Extracting waveforms and peak times
            print(str(len(waveforms_chunk)) + ' spikes found in the chunk.' )

            waveforms = np.append(waveforms, waveforms_chunk, 0) #Appending the waveforms from the chunk to the waveforms for the location
            peak_times = np.append(peak_times, peak_times_chunk + current_time*experiment.sample_rate + reading_index) #Appending the peak times from the chunk to the peak times for the location

            data_all = np.append(data_all, data, 1)
            reading_index = reading_index + chunk_size
        current_time = current_time + time_session[-1] + 1/experiment.sample_rate

    location.time = time
    location.end_inds = end_inds
    location_output = {'waveforms': waveforms, 'peak_times':peak_times, 'time':time, 'end_inds':end_inds, 'data_all':data_all}

    return location_output

def extract_waveforms(filtered, time, location,noise):
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
    thresh = noise * experiment.threshold_coeff
    peak_times = []

    for trode in range(len(filtered)):
        peak_times_trode = signal.find_peaks(-filtered[trode], height = thresh[trode])[0]
        peak_times.append(peak_times_trode)

    peak_times = np.asarray(peak_times)[0]
    peak_times = np.sort(peak_times)

    waveforms = np.zeros((len(peak_times), len(filtered), experiment.spike_samples_before+experiment.spike_samples_after))
    for i, spike in enumerate(peak_times):
        if (spike > experiment.spike_samples_before) and (spike < len(time) - experiment.spike_samples_after):
            waveforms[i] = filtered[:,(spike-experiment.spike_samples_before):(spike+experiment.spike_samples_after)]

    return waveforms, peak_times
