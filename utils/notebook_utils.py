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
    spike_timerange = np.arange(-experiment.spike_samples_before, experiment.spike_samples_after, 1) / (experiment.sample_rate/1000)
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
    spike_timerange = np.arange(-experiment.spike_samples_before, experiment.spike_samples_after, 1) / (experiment.sample_rate/1000)

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
