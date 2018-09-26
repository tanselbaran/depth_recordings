
# coding: utf-8

# # Surface LFP Analysis Pipeline 
# 
# Welcome to the script for generating parameter dictionaries for the recording sessions in your experiment folder. Please follow the upcoming steps in this notebook for further instructions. 
# 
# ## 1) Import the packages required for running the script
# 
# Please run the block of code to import the Python packages that are required for running the rest of this script. 

# In[1]:


#Import required packages
import pickle as p 
import os
from utils.load_intan_rhd_format import * 
from utils.reading_utils import *
from tqdm import tqdm
import ipywidgets
from ipywidgets import Layout, HBox, VBox
from IPython.display import display
from glob import glob
import importlib
from utils.experiment_classes import *
from utils.filtering import * 
import numpy as np
import h5py
from spikeSortingUtils.custom_spike_sorting_utils import * 
from utils.notebook_utils import * 


# ## 2) Enter general parameters for the experiment.

# In[ ]:


#Creating widgets for the user input on the general parameters for the experiment

##Main path for the data 

"""mp_html = ipywidgets.HTML(value = "<p><b>Path to the data of the experiment:</b><br />Enter the path to the folder (with no '/' at the end) that is hierarchically right above the folders of the recording sessions</p>")
mp = ipywidgets.Text(value = "", placeholder = "Enter path for data", disabled = False)
display(VBox([mp_html, mp]))

##Type of the experiment

et_html = ipywidgets.HTML(value = "<b>Type of the experiment</b>")
et = ipywidgets.Dropdown(options=['Acute', 'Chronic'], 
                   value = 'Acute',  disabled = False)
display(VBox([et_html, et]))


##File format
ff_html = ipywidgets.HTML(value = "<p><b>File format:</b><br />(dat for .dat, cont for .continuous, rhd for .rhd)</p>")
ff = ipywidgets.Text(value = 'dat', placeholder = 'Enter file format',
             disabled = False)
display(VBox([ff_html,ff]))

##Number of probes

nump_html = ipywidgets.HTML(value = "<p><b>Number of probes:</b><br /><b>WARNING:</b>Pipeline currently supports <b>ONLY</b> the multiple probes being <b>IDENTICAL</b> in type and mapping!!! Pipeline has to be updated before using multiple probes of different types!</p>")
nump = ipywidgets.IntText(value = 1, disabled = False)
display(VBox([nump_html, nump]))

##Probe info
pi_html = ipywidgets.HTML(value = "<b>Probe used in the experiment</b>")
pi = ipywidgets.Dropdown(options=['thirty_channel_ecog', 'thirtytwo_channel_ecog', 'a4x8_5mm_100_200_177', 'a4x4_tet_150_200_1212', 'a2x16_10mm_100_500_177'], 
                   value = 'a2x16_10mm_100_500_177',  disabled = False)
display(VBox([pi_html, pi]))

##Amplifier port
ap_html = ipywidgets.HTML(value = "<b>The port to which the amplifier is connected</b>")
ap = ipywidgets.Dropdown(options=['A', 'B', 'C', 'D'], 
                   value = 'A',  disabled = False)
display(VBox([ap_html, ap]))

##Dead channels 
dc_html = ipywidgets.HTML(value = "<p><b>Please enter dead channels with commas in between (e.g. 1,5,10,23) </b></p>")
dc = ipywidgets.Text(value = '', placeholder = 'Enter dead channels',
             disabled = False)
display(VBox([dc_html,dc]))

##Whisker Stim Path

wsp = ipywidgets.IntText(value = 1, disabled = False)
display(VBox([ipywidgets.HTML(value = "<b>Enter the index of the digital input channel where the whisker stimulation trigger is kept</b>"), wsp]))

##Optical Stim Path

osp = ipywidgets.IntText(value = 1, disabled = False)
display(VBox([ipywidgets.HTML(value = "<b>Enter the index of the digital input channel where the optical stimulation trigger is kept</b>"), osp]))

##Electrical Stim Path

esp = ipywidgets.IntText(value = 1, disabled = False)
display(VBox([ipywidgets.HTML(value = "<b>Enter the index of the digital input channel where the electrical stimulation trigger is kept</b>"), esp]))


# ## 3) Enter parameters related to evoked LFP analysis

# In[ ]:


#Creating widgets for the user input on the parameters related to the whisker stim evoked LFP analysis

##Downsampling factor
ds = ipywidgets.IntText(value = 30, disabled = False)
display(VBox([ipywidgets.HTML(value = "<b>Enter the factor of downsampling the raw data prior to evoked LFP analysis. </b>"), ds]))

##whiskerEvokedPre

wpre = ipywidgets.FloatText(value = 0.025, disabled = False)
display(VBox([ipywidgets.HTML(value = "<b>Enter time taken prior to the whisker stimulus trigger (in s)</b>"), wpre]))

##whiskerEvokedPost

wpost = ipywidgets.FloatText(value = 0.100, disabled = False)
display(VBox([ipywidgets.HTML(value = "<b>Enter time taken post the whisker stimulus trigger (in s)</b>"), wpost]))

##lightEvokedPre

lpre = ipywidgets.FloatText(value = 0.025, disabled = False)
display(VBox([ipywidgets.HTML(value = "<b>Enter time taken prior to the optical stimulus trigger (in s)</b>"), lpre]))

##lightEvokedPost

lpost = ipywidgets.FloatText(value = 0.100, disabled = False)
display(VBox([ipywidgets.HTML(value = "<b>Enter time taken post the optical stimulus trigger (in s)</b>"), lpost]))

#low_pass_freq

lp = ipywidgets.FloatText(value = 300, disabled = False)
display(VBox([ipywidgets.HTML(value = "<b> Enter the cutoff frequency of the low pass filter to extract LFP from data (in Hz)"), lp]))

#Low pass filter order
lp_order = ipywidgets.IntText(value = 4, disabled = False)
display(VBox([ipywidgets.HTML(value = "<b> Enter the order of the Butterworth low pass filter used for the evoked LFP analysis (in Hz)"), lp_order]))

#notch_filt_freq

nf = ipywidgets.FloatText(value = 0, disabled = False)
display(VBox([ipywidgets.HTML(value = "<b> Enter the frequency of the notch filter (in Hz). Enter 0 if you don't want a notch filter"), nf]))

##cutBeginning

cb = ipywidgets.FloatText(value = 1, disabled = False)
display(VBox([ipywidgets.HTML(value = "<b>Enter the time to be cut from the beginning of the recording (in s)</b>"), cb]))

##cutEnd

ce = ipywidgets.FloatText(value = 1, disabled = False)
display(VBox([ipywidgets.HTML(value = "<b>Enter the time to be cut from the end of the recording (in s )"), ce]))



# # 4) Enter parameters related to spike sorting
# 
# If you are intending to do spike-sorting on this data, please set the spike-sorting parameters. Otherwise, set the boolean parameter "do_spike_sorting" to False below. 

# In[ ]:


#Creating widgets for the user input on the parameters related to spike sorting

##samplesBefore

sb = ipywidgets.FloatText(value = 0.5, disabled = False)
display(VBox([ipywidgets.HTML(value = '<b>Enter the length of waveform to be taken before the threshold crossing (in ms)'), sb]))

##samplesAfter

sa = ipywidgets.FloatText(value = 1.5, disabled = False)
display(VBox([ipywidgets.HTML(value = "<b>Enter the length of waveform to be taken after the threshold crossing (in ms)"), sa]))

##lowCutoff

lc = ipywidgets.FloatText(value = 500., disabled = False)
display(VBox([ipywidgets.HTML(value = "<b>Enter the lower cutoff frequency for the bandpass filter to be applied on the raw data (in Hz)"), lc]))

##highCutoff

hc = ipywidgets.FloatText(value = 5000., disabled = False)
display(VBox([ipywidgets.HTML(value = "<b>Enter the higher cutoff frequency for the bandpass filter to be applied on the raw data (in Hz)"), hc]))

##chunkSize

cs = ipywidgets.IntText(value = 60, disabled = False)
display(VBox([ipywidgets.HTML(value = "<b>Enter the chunk size (in s) for when reading the raw data to be processed for spike detection and sorting"), cs]))

##thresholdingCoefficient

tc = ipywidgets.FloatText(value = 4.5, disabled = False)
display(VBox([ipywidgets.HTML(value = "<b>Enter the thresholding coefficient (in terms of multiple of baseline noise rms) to be used for spike detection"), tc]))

##minFrac

mf = ipywidgets.FloatText(value = 0.01, disabled = False)
display(VBox([ipywidgets.HTML(value = "<b>Enter the minimum variance described by the principal components while applying PCA on the band-pass filtered data"), mf]))

##numClusters

nc_html = ipywidgets.HTML(value = "<b>Enter the number of initial seeds used for k-means clustering (three numbers separated by comma; one each for initial clustering, good clustering and better clustering)")
nc = ipywidgets.Text(value = '200,100,100', placeholder = 'Enter number of clusters',
             disabled = False)
display(VBox([nc_html,nc]))

##Bandpass filter order 

fo = ipywidgets.IntText(value = 6)
display(VBox([ipywidgets.HTML(value = "<b>Enter the order of the Butterworth bandpass filter used on the raw data"), fo]))


# # 5) Generate the parameters dictionaries
# 
# Please run the block of the code in order to generate the parameters dictionary for each recording session (paramsDict.p) based on the input that you have provided above. 

# In[ ]:


dead_channels = np.asarray(dc.value.split(','))
dead_channels = dead_channels.astype('int8')

if et.value == 'Acute':
    experiment = acute(mp.value)
    locations = glob(mp.value + '/loc*')
    
    for location in locations:
        experiment.add_location(location)
        
    for location_index in experiment.locations:
        location = experiment.locations[location_index]
        location.add_sessions_in_dir()
        
        for session_index in location.sessions:
            session = location.sessions[session_index]
            session.setTrigChannels(wsp.value, osp.value, esp.value)
            session.createProbe(pi.value)
            session.probe.remove_dead_channels(dead_channels)
        
    amplifier = experiment.locations[0].sessions[0].amplifier
    if amplifier == 'rhd':
        header_dir = experiment.locations[0].sessions[0].dir + '/info.rhd' 
            
        
elif et.value == 'Chronic':
    experiment = chronic(mp.value)
    days = glob(mp.value + '/20*')
    
    for day in days:
        experiment.add_day(day)
        
    for day_index in experiment.days:
        day = experiment.days[day_index]
        day.add_sessions_in_dir()
        
        for session_index in day.sessions:
            session = day.sessions[session_index]
            session.setTrigChannels(wsp.value, osp.value, esp.value)
            session.createProbe(pi.value)
            session.probe.remove_dead_channels(dead_channels)
    
    amplifier = experiment.days[0].sessions[0].amplifier
    if amplifier == 'rhd':
        header_dir = experiment.days[0].sessions[0].dir + '/info.rhd'

if amplifier == 'rhd':
    header = read_data(header_dir)
    sr = header['frequency_parameters']['amplifier_sample_rate']
else: 
    sr=30000

#General parameters   
experiment.fileformat = ff.value
experiment.sample_rate = sr

#Parameters related to spike sorting 
experiment.low_cutoff_bandpass = lc.value
experiment.high_cutoff_bandpass = hc.value
experiment.spike_samples_before = int(sb.value * experiment.sample_rate / 1000)
experiment.spike_samples_after = int(sa.value * experiment.sample_rate / 1000)
experiment.chunk_size = cs.value 
experiment.threshold_coeff = tc.value 
experiment.min_frac = mf.value 
num_clusters = nc.value.split(',')
num_clusters = np.asarray(num_clusters)
experiment.num_clusters = num_clusters.astype('int8')
experiment.bandfilter_order = fo.value
experiment.bandfilt = bandpassFilter(rate = experiment.sample_rate, high = experiment.high_cutoff_bandpass, low = experiment.low_cutoff_bandpass, order = experiment.bandfilter_order, axis = 1)

#Parameters related to evoked LFP analysis 
experiment.cut_beginning = cb.value 
experiment.cut_end = ce.value
experiment.low_pass_freq = lp.value
experiment.low_pass_order = lp_order.value 
experiment.notch_filt_freq = nf.value
experiment.whisker_evoked_pre = wpre.value
experiment.whisker_evoked_post = wpost.value
experiment.optical_evoked_pre = lpre.value
experiment.optical_evoked_post = lpost.value
experiment.downsampling_factor = ds.value 


# In[ ]:


pickle.dump(experiment, open((mp.value + '/experiment_params.p'), 'wb'))


# # 6) Read and analyze stimulus evoked LFPs, generate dictionaries for spike sorting analysis

# In[2]:


#f.close()"""
experiment = pickle.load(open('/home/baran/Dropbox (Yanik Lab)/Layer 1 Project/Electrophysiology/Experiments/20180709_BY17_opto_ephys/experiment_params.p', 'rb'))

f = h5py.File(experiment.dir + '/analysis_results.hdf5', 'a')
experiment.probe = experiment.locations[0].sessions[0].probe

for location_index in experiment.locations:
    location = experiment.locations[location_index]
    print("Location: " + location.name)
    loc_grp = f.create_group(location.name)
    
    for group in (range(experiment.probe.nr_of_groups)):
        if not os.path.exists(location.dir + '/analysis_files'):
            os.mkdir(location.dir + '/analysis_files/')
        if not os.path.exists(location.dir + '/analysis_files/group_{:g}'.format(group)):
            os.mkdir(location.dir + '/analysis_files/group_{:g}'.format(group))
        if location.preferences['do_spike_analysis'] == 'y':
            initialize_spike_sorting_notebook_for_group(location_index, location, group)
            
    for session_index in location.sessions: 
        session = location.sessions[session_index]
        print("Session: " + session.name)
        ses_grp = loc_grp.create_group(session.name)
        if (session.preferences['do_whisker_stim_evoked'] == 'y') or (session.preferences['do_optical_stim_evoked'] == 'y'):
            stim_timestamps = read_stimulus_trigger(session)
            for group in (range(experiment.probe.nr_of_groups)):
                print("Channel group: " + str(group))
                ch_grp = ses_grp.create_group('group_{:g}'.format(group))
                read_evoked_lfp(group,session)               
f.close()


# Notebook written by Baran Yasar in 04/2017. Please contact him in person or e-mail at yasar@biomed.ee.ethz.ch in case of any questions. 
# 
# ---Updated in 07/2018 by Baran Yasar
