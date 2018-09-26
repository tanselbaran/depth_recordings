
# coding: utf-8

# # Notebook for analyzing location: loc1,  group: 0 in experiment: 20180709_BY17_opto_ephys
#             

# ## Importing necessary packages and initializing objects

# In[1]:


import os
os.chdir('/home/baran/Desktop/git-repos/surface_recording_project')

from spikeSortingUtils.custom_spike_sorting_utils import *
import numpy as np
import pickle as p
from utils.notebook_utils import *
from spikeSortingUtils.simple_clustering_utils import *
import h5py

get_ipython().run_line_magic('matplotlib', 'notebook')

experiment_main_folder = '/home/baran/Dropbox (Yanik Lab)/Layer 1 Project/Electrophysiology/Experiments/20180709_BY17_opto_ephys'
experiment = p.load(open(experiment_main_folder + '/experiment_params.p','rb'))
location = experiment.locations[2]
analysis_files_folder = location.dir+'/analysis_files/group_0/'


# ## Reading the data and extracting waveforms from this location

# In[2]:


location_output = read_location(location, 0)
clusters, projection = PCA_and_cluster(location_output['waveforms'], location)
save_clusters_to_pickle(clusters, projection, location_output['peak_times'], (analysis_files_folder+'cluster_info.p'))
display_widget(location_output['waveforms'], plot_params, location, ['clusters', 'ind_on'], clusters)


# ## Reclustering the selected waveforms

# In[ ]:


good_cluster_indices = []
(good_clusters, good_waveforms, good_peaktimes, good_projection) = recluster(waveforms, location, peak_times, clusters, good_cluster_indices)
save_reclusters_to_pickle(good_clusters, good_projection, good_peaktimes, (analysis_files_folder+'good_cluster_info.p'))
display_widget(good_waveforms, plot_params, location, ['clusters', 'ind_on'], good_clusters)


# ## Reclustering selected waveforms again

# In[ ]:


better_cluster_indices = []
(better_clusters, better_waveforms, better_peaktimes, better_projection) = recluster(good_waveforms, location, good_peaktimes, good_clusters, better_cluster_indices)
del good_waveforms
save_reclusters_to_pickle(better_clusters, better_projection, better_peaktimes, (analysis_files_folder+'better_cluster_info.p'))
display_widget(better_waveforms, plot_params, location, ['clusters', 'ind_on'], better_clusters)


# ## Sorting into units

# In[ ]:


noise = []
multi_unit = []
units = {
    0:[],
    1:[],}
unit_indices = get_unit_indices(units, better_clusters)
spike_times, spike_trains = get_unit_spike_times_and_trains(unit_indices, better_peaktimes, location)
spike_trains_location = break_down_to_sessions(location, spike_times, spike_trains)


# ## Generating PSTHs for sessions based on provided stim preferences

# In[ ]:


generate_psths(location, spike_times)


# ## Saving the spike trains, waveforms and PSTHs in hdf file

# In[ ]:


f = h5py.File(experiment_main_folder + '/analysis_results.hdf5', 'a')
ch_grp = f[location.name + '/group_0']
spike_grp = ch_grp.create_group('spike_results')
spike_grp.create_dataset("spike_trains", spike_trains)
spike_grp.create_dataset("spike_times", spike_times)
spike_grp.create_dataset("unit_indices", unit_indices)
spike_grp.create_dataset("waveforms", better_waveforms)
if whisker_stim_psth in locals():
    spike_grp.create_dataset("whisker_stim_psth",whisker_stim_psth)
if optical_stim_psth in locals():
    spike_grp.create_dataset("optical_stim_psth", optical_stim_psth)

