import numpy as np
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D
from utils.filtering import *
from utils.load_intan_rhd_format import *
from utils.reading_utils import *
from tqdm import tqdm
from LFPutils.read_evoked_lfp import *
import pickle as p
from ipywidgets import interact, IntText, fixed, FloatSlider
from IPython.display import display
import nbformat as nbf


def initialize_spike_sorting_notebook_for_group(location_index, location, group):
    nb = nbf.v4.new_notebook()
    experiment = location.experiment

    header = """# Notebook for analyzing location: {:s},  group: {:g} in experiment: {:s}
            """.format(location.name, group, experiment.name)

    import_header = """## Importing necessary packages and initializing objects"""
    import_code = """import os
os.chdir('/home/baran/Desktop/git-repos/surface_recording_project')

from spikeSortingUtils.custom_spike_sorting_utils import *
import numpy as np
import pickle as p
from utils.notebook_utils import *
from spikeSortingUtils.simple_clustering_utils import *
import h5py

%matplotlib notebook

experiment_main_folder = '{:s}'
experiment = p.load(open(experiment_main_folder + '/experiment_params.p', 'rb'))
location = experiment.locations[{:g}]
analysis_files_folder = location.dir+'/analysis_files/group_{:g}/'""".format(experiment.dir, location_index, group)

    read_location_header = """## Reading the data and extracting waveforms from this location"""
    read_location_code = """location_output = read_location(location, {:g})
clusters, projection = PCA_and_cluster(location_output['waveforms'], location, {:g}, {:g})
save_clusters_to_pickle(clusters, projection, location_output['peak_times'], (analysis_files_folder+'cluster_info.p'))
display_widget(location_output['waveforms'], plot_params, location, ['clusters', 'ind_on'], clusters)""".format(group, experiment.min_frac, experiment.num_clusters[0])

    good_reclustering_header = """## Reclustering the selected waveforms"""
    good_reclustering_code = """good_cluster_indices = []
(good_clusters, good_waveforms, good_peaktimes, good_projection) = recluster(location_output['waveforms'], location, location_output['peak_times'], {:g}, {:g}, clusters, good_cluster_indices)
save_reclusters_to_pickle(good_clusters, good_waveforms, good_projection, good_peaktimes, (analysis_files_folder+'good_cluster_info.p'))
display_widget(good_waveforms, plot_params, location, ['clusters', 'ind_on'], good_clusters)""".format(experiment.min_frac, experiment.num_clusters[1])

    better_reclustering_header = """## Reclustering selected waveforms again"""
    better_reclustering_code = """better_cluster_indices = []
(better_clusters, better_waveforms, better_peaktimes, better_projection) = recluster(good_waveforms, location, good_peaktimes, {:g}, {:g}, good_clusters, better_cluster_indices)
del good_waveforms
save_reclusters_to_pickle(better_clusters, better_waveforms, better_projection, better_peaktimes, (analysis_files_folder+'better_cluster_info.p'))
display_widget(better_waveforms, plot_params, location, ['clusters', 'ind_on'], better_clusters)""".format(experiment.min_frac, experiment.num_clusters[2])

    sorting_header = """## Sorting into units"""
    sorting_code = """noise = []
multi_unit = []
units = {
    0:[],
    1:[],}
unit_indices = get_unit_indices(units, better_clusters)
spike_times, spike_trains = get_unit_spike_times_and_trains(unit_indices, better_peaktimes, location)
spike_trains_location = break_down_to_sessions(location, spike_times, spike_trains)"""

    psth_header = """## Generating PSTHs for sessions based on provided stim preferences"""
    psth_code = """generate_psths(location, spike_times)"""

    saving_header = """## Saving the spike trains, waveforms and PSTHs in hdf file"""
    saving_code = """f = h5py.File(experiment_main_folder + '/analysis_results.hdf5', 'a')
ch_grp = f[location.name + '/group_{:g}']
spike_grp = ch_grp.create_group('spike_results')
spike_grp.create_dataset("spike_trains", spike_trains)
spike_grp.create_dataset("spike_times", spike_times)
spike_grp.create_dataset("unit_indices", unit_indices)
spike_grp.create_dataset("waveforms", better_waveforms)
if whisker_stim_psth in locals():
    spike_grp.create_dataset("whisker_stim_psth",whisker_stim_psth)
if optical_stim_psth in locals():
    spike_grp.create_dataset("optical_stim_psth", optical_stim_psth)""".format(group, )

    nb['cells'] = [nbf.v4.new_markdown_cell(header),
                        nbf.v4.new_markdown_cell(import_header),
                        nbf.v4.new_code_cell(import_code),
                        nbf.v4.new_markdown_cell(read_location_header),
                        nbf.v4.new_code_cell(read_location_code),
                        nbf.v4.new_markdown_cell(good_reclustering_header),
                        nbf.v4.new_code_cell(good_reclustering_code),
                        nbf.v4.new_markdown_cell(better_reclustering_header),
                        nbf.v4.new_code_cell(better_reclustering_code),
                        nbf.v4.new_markdown_cell(sorting_header),
                        nbf.v4.new_code_cell(sorting_code),
                        nbf.v4.new_markdown_cell(psth_header),
                        nbf.v4.new_code_cell(psth_code),
                        nbf.v4.new_markdown_cell(saving_header),
                        nbf.v4.new_code_cell(saving_code)]

    nbf.write(nb, location.dir+'/analysis_files/group_{:g}/spike_sorting_notebook.ipynb'.format(group))
### Utilities for processing the units

def get_unit_indices(units, clusters):
    unit_indices = {}
    for unit in range(len(units)):
        unit_idx = np.zeros(0)
        for cluster in range(len(units[unit])):
            cluster_indices = np.where(clusters.labels_ == units[unit][cluster])
            unit_idx = np.append(unit_idx, cluster_indices)
        unit_idx = unit_idx.astype('int')
        unit_indices[unit] = unit_idx

    return unit_indices

def get_unit_spike_times_and_trains(unit_indices, time, peak_times, location):
    spike_times = {}
    spike_trains = np.zeros((len(unit_indices), len(time)))
    spike_trains = spike_trains.astype('int8')

    for unit in range(len(unit_indices)):
        spike_times_ind = peak_times[unit_indices[unit]]
        spike_times_ind = np.asarray(spike_times_ind * location.experiment.sample_rate)
        spike_times_ind = spike_times_ind.astype(int)
        spike_times[unit] = np.sort(spike_times_ind)
        spike_trains[unit][spike_times_ind] = 1

    return spike_times, spike_trains

### Utilites for plotting

def plot_waveforms(index, waveforms, plot_params, location):
    """
    This function serves as the interactive function for the widget for displaying waveforms across multiple channels.

    Inputs:
        index: Index of the waveform to be displayed [int]
        waveforms: Numpy array containing the waveforms; in the form of (N_events x N_electrodes x N_spike_time_range_steps)
        mapping: Mapping attribute in the h5 data
        plot_params: Dictionary containing parameters related to plotting. Must contain following entries:
            nrow: Number of rows in the subplot grid
            ncol: Number of columns in the subplot grid
            ylim: Limits of the y axis for all electrodes, in the form of [ymin, ymax]
        params: Dictionary containing recording parameters
        flat_map_array: Flattened numpy array containing the spatial mapping information of the electrodes
            spike_timerange: Array containing the time range of the spike time window
    """

    fig, axs = subplots(plot_params['nrow'], plot_params['ncol'])
    channel = 0
    experiment=location.experiment
    spike_timerange = np.arange(-experiment.spike_samples_before, experiment.spike_samples_after, 1) / experiment.sample_rate
    for i, ax in enumerate(fig.axes):
        ax.plot(spike_timerange, waveforms[index, channel])
        ax.set_ylim(plot_params['ylim'])
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Voltage (uV)')
        channel = channel+1
    show()

def plot_mean_cluster_waveforms(cluster, clusters, waveforms, plot_params, location, mode):
    """
    This function serves as the interactive function for the widget for displaying waveforms across multiple channels.

    Inputs:
        cluster: Index of the cluster for which the waveforms to be displayed [int]
        clusters: Result of the k-means clustering on the projection of the waveforms on the principal component axes
        waveforms: Numpy array containing the waveforms; in the form of (N_events x N_electrodes x N_spike_time_range_steps)
        mapping: Mapping attribute in the h5 data
        plot_params: Dictionary containing parameters related to plotting. Must contain following entries:
            nrow: Number of rows in the subplot grid
            ncol: Number of columns in the subplot grid
            ylim: Limits of the y axis for all electrodes, in the form of [ymin, ymax]
        params: Dictionary containing recording parameters
            flat_map_array: Flattened numpy array containing the spatial mapping information of the electrodes
            spike_timerange: Array containing the time range of the spike time window
        mode: Plot the mean waveform with or without the individual waveforms ('ind_on' for displaying the individual waveforms, 'ind_off' for not displaying them)
    """
    experiment=location.experiment
    spike_timerange = np.arange(-experiment.spike_samples_before, experiment.spike_samples_after, 1) / experiment.sample_rate

    #Selecting the spike waveforms that belong to the selected cluster and calculating the mean waveform at each electrode
    spikes_in_cluster = waveforms[np.where(clusters.labels_ == cluster)]
    mean_spikes_in_cluster = np.mean(spikes_in_cluster, 0)

    fig, axs = subplots(plot_params['nrow'], plot_params['ncol'])
    channel = 0
    for i, ax in enumerate(fig.axes):
        ax.plot(spike_timerange, mean_spikes_in_cluster[channel])
        if mode == 'ind_on':
            for spike in range(len(spikes_in_cluster)):
                ax.plot(spike_timerange, spikes_in_cluster[spike,i], 'b', alpha = 0.1)
        ax.set_ylim(plot_params['ylim'])
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Voltage (uV)')
        channel = channel + 1
    show()

def plot_3d_of_clusters(clusters, projection, location):
    """
    This function makes a 3d scatter plot of the projections of the spike waveforms  onto the 3 principal axes that count for the highest fraction of variance in the original waveforms array. The scatter points are colored with respect to the clusters that each spike waveform belongs to.

    Inputs:
        clusters: KMeans object that contains information about the results of the K-means clustering of the waveforms projected on the principal component axes
        projection: 2d numpy array (# of spike_events x # of PCA axes) that contains the projections of the spike waveforms on the PCA axes
    """

    fig = figure()
    ax = fig.add_subplot(111, projection = '3d')
    for cluster in range(clusters.n_clusters):
        cluster_indices = np.where(clusters.labels_ == cluster)
        ax.scatter(projection[:,0][cluster_indices], projection[:,1][cluster_indices], projection[:,2][cluster_indices], location.experiment.colors[cluster])
    show()

### IPython widget utilities for interactivity

def display_widget(waveforms, plot_params, location, mode, *args):
    """
    This function creates and displays the widget for selecting the waveform or cluster index to be displayed and the widget that displays
    the waveforms or mean waveforms for the clusters across multiple channels for this index.

    Inputs:
        waveforms: Numpy array containing the waveforms; in the form of (N_events x N_electrodes x N_spike_time_range_steps )
        plot_params: see above
        mapping: Mapping attribute in the h5 data
        params: see above
        mode: 'waveforms' for displaying individual waveforms by index, 'clusters' for displaying the mean waveform of a cluster
    """
    if mode[0] == 'waveforms':
        #Widget for selecting waveform or cluster index
        selectionWidget = IntText(min=0, max=len(waveforms), step = 1, value = 0,
            description = "Waveforms to be displayed", continuous_update = True)

        #Widget for plotting selected waveforms or mean waveforms
        widget = interact(plot_waveforms, index = selectionWidget,
            waveforms = fixed(waveforms), plot_params = fixed(plot_params), location = fixed(location))
    elif mode[0] == 'clusters':
        #Widget for selecting waveform or cluster index
        selectionWidget = IntText(min=0, max=len(waveforms), step = 1, value = 0,
            description = "Cluster for which the waveforms to be displayed", continuous_update = True)

        #Widget for plotting selected waveforms or mean waveforms
        widget = interact(plot_mean_cluster_waveforms, cluster = selectionWidget,
            clusters = fixed(args[0]), waveforms = fixed(waveforms),
            plot_params = fixed(plot_params), location = fixed(location), mode = mode[1])
    else:
        raise ValueError('Please select a valid mode for display ("waveforms" or "clusters")')

    display(selectionWidget)
    display(widget)

### Utilities for long-term storage and retrieval of the analysis results

def save_clusters_to_pickle(clusters, projection, peak_times, filepath):
    p.dump({'clusters': clusters, 'projection': projection}, open(filepath, 'wb'))

def save_reclusters_to_pickle(clusters, waveforms, projection, peak_times, filepath):
    p.dump({'clusters': clusters, 'waveforms': waveforms, 'peak_times': peak_times, 'projection': projection}, open(filepath, 'wb'))

def read_clusters_from_pickle(filepath):
    cluster_dict = p.load(open(filepath, 'rb'))
    clusters = cluster_dict['clusters']
    projection = cluster_dict['projection']
    return clusters, projection

def read_reclusters_from_pickle(filepath):
    cluster_dict = p.load(open(filepath, 'rb'))
    clusters = cluster_dict['clusters']
    waveforms = cluster_dict['waveforms']
    peak_times = cluster_dict['peak_times']
    projection = cluster_dict['projection']
    return clusters, waveforms, peak_times, projection

def save_waveforms_to_pickle(waveforms, peak_times, filepath):
    p.dump({'waveforms': waveforms, 'peak_times': peak_times}, open(filepath, 'wb'))

def read_waveforms_from_pickle(filepath):
    waveforms_dict = p.load(open(filepath, 'rb'))
    waveforms = waveforms_dict['waveforms']
    peak_times = waveforms_dict['peak_times']
    return waveforms, peak_times
