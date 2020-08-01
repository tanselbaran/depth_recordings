"""
Created on Friday, Oct 6th, 2017

author: Tansel Baran Yasar

Contains the functions for assessing the quality of clustering.
"""

import numpy as np
from matplotlib.pyplot import *
import math
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

def get_spike_features_and_unit_ids(session):
	"""
	This function extracts the coordinates of each spike in the session from the kwx file in the feature space determined by the dimensionality reduction done by Klusta (N_electrodes x 3 features).

	Inputs:
	 	session: Session object to be analyzed

	Outputs:
		features: N_spikes x (N_electrodes x 3) numpy array containing the coordinates of each spike in the feature space.
	"""
	experiment = session.subExperiment.experiment
    spike_sorting_folder = experiment.dir + '/preprocessing_files/' + session.subExperiment.name + '/' + session.name + '/spike_sorting/group_0/'

    kwx_file = h5py.File(spike_sorting_folder + 'group_0.kwx', 'r')
	kwik_file = h5py.File(spike_sorting_folder + 'group_0.kwik', 'r')
	preprocessing_file = h5py.File(experiment.dir + 'preprocessing_results.hdf5', 'r')

	cluster_id = np.asarray(kwik_file['channel_groups/0/spikes/clusters/main'])
	units = np.asarray(list(preprocessing_file[session.subExperiment.name + '/' + session.name + '/group_0'].keys()), dtype='int16')

	features_masks = np.asarray(kwx_file['channel_groups/0/features_masks'])
	features = features_masks[:,:,0]

	kwx_file.close()
	kwik_file.close()
	preprocessing_file.close()

	return features, units, cluster_id

def get_spike_mahalanobis_distances(session):
	"""
	This function calculates the Mahalanobis distance of each spike from each of the unit cluster center of masses in the feature space (based on the dimensionality reduction done by Klusta)

	Inputs:

	"""
	features, units, cluster_id = get_spike_features_and_unit_ids(session)
	spk_mah_dists = np.zeros((len(cluster_id), len(units)))

	for i, unit in enumerate(units):
		unit_spikes = np.where(cluster_id == unit)[0]
		unit_spike_coords = fe.atures[unit_spikes]
		unit_center_of_mass = np.mean(unit_spike_coords, 0)
		unit_inv_cov = np.linalg.inv(np.cov(unit_spike_coords, rowvar=False))
		for spike in range(len(cluster_id)):
			spk_mah_dists[spike,i] = mahalanobis(features[spike], unit_center_of_mass, unit_inv_cov)

	preprocessing_file.close()

	return spk_mah_dists

def L_ratio(session):
	preprocessing_file = h5py.File(experiment.dir + 'preprocessing_results.hdf5', 'r+')
	features, units, cluster_id = get_spike_features_and_unit_ids(session)
	spk_mah_dists = get_spike_mahalanobis_distances(session)

	L_ratios = np.zeros(len(units))

	for i,unit in enumerate(units):
		unit_spikes = np.where(cluster_id == unit)[0]
		non_unit_spikes = np.where(cluster_id != unit)[0]
		cdf = chi2.cdf(np.square(spk_mah_dists[noise_spikes, i]), len(features[0]))
		L_ratio = np.sum(1-cdf)/len(unit_spikes)
		L_ratios[i] = L_ratio

		unit_grp = preprocessing_file[session.subExperiment.name + '/' + session.name + '/group_0/' + str(unit)]
		unit_grp.create_dataset("L_ratio", data=L_ratios[i])

	return L_ratios

def isolation_distance(session):
	"""
	Isolation distance between each unit from "noise" is calculated as defined in Harris et al 2001, Neuron paper.

	Inputs:
		session: Session object for the recording session to be analyzed

	Outputs:
		isolation_distances: N_units x 1 numpy array containing the Isolation Distance of each unit cluster identified in the recording session from spike sorting
	"""
	preprocessing_file = h5py.File(experiment.dir + 'preprocessing_results.hdf5', 'r+')
	features, units, cluster_id = get_spike_features_and_unit_ids(session)
	spk_mah_dists = get_spike_mahalanobis_distances(session)

	isolation_distances = np.zeros(len(units))
	for i,unit in enumerate(units):
		unit_spikes = np.where(cluster_id == unit)[0]
		non_unit_spikes = np.where(cluster_id != unit)[0]
		spike_count = len(unit_spikes)
		isolation_distances[i] = np.square(np.sort(spk_mah_dists[non_unit_spikes, i])[spike_count])

		unit_grp = preprocessing_file[session.subExperiment.name + '/' + session.name + '/group_0/' + str(unit)]
		unit_grp.create_dataset("isolation_distance", data=isolation_distances[i])

	return isolation_distances

def calculate_unit_distances(session):
	preprocessing_file = h5py.File(experiment.dir + 'preprocessing_results.hdf5', 'r+')
	features, units, cluster_id = get_spike_features_and_unit_ids(session)
	unit_cms = np.zeros((len(units), len(features[0])))
	euc = np.zeros((len(units), len(units)))
	maho = np.zeros((len(units), len(units)))

	inv_cov = np.linalg.inv(np.cov(features, rowvar=False))

	for i, unit in enumerate(units):
		unit_spikes = np.where(cluster_id == unit)[0]
		unit_spike_coords = features[unit_spikes]
		unit_cms[i] = np.mean(unit_spike_coords, 0)

	for i in range(len(units)):
		for j in range(len(units)):
			if j==i:
				euc[i,j] = np.linalg.norm(unit_cms[i] - np.zeros(len(features[0])))
				maho[i,j] = mahalanobis(unit_cms[i], np.zeros(len(features[0])), inv_cov)
			else:
				euc[i,j] = np.linalg.norm(unit_cms[i] - unit_cms[j])
				maho[i,j] = mahalanobis(unit_cms[i], unit_cms[j], inv_cov)

	group_grp = preprocessing_file[session.subExperiment.name + '/' + session.name + '/group_0/']
	group_grp.create_dataset("euclidian", data=euc)
	group_grp.create_dataset("mahalanobis", data=maho)
	return euc, maho

def ISI_violations(session, cutoff):
	experiment = session.subExperiment.experiment
	preprocessing_file = h5py.File(experiment.dir + 'preprocessing_results.hdf5', 'r+')
	features, units, cluster_id = get_spike_features_and_unit_ids(session)
	units = np.asarray(list(preprocessing_file[session.subExperiment.name + '/' + session.name + '/group_0'].keys()), dtype='int16')

	ISI_violations = np.zeros(len(units))

	for i, unit in enumerate(units):
		spike_times = np.asarray(preprocessing_file[session.subExperiment.name + '/' + session.name + '/group_0/' + str(unit)], dtype='int16')
		ISI[i] = (np.diff(spike_times) / experiment.sample_rate) * 1000
		ISI_violation_instances = ISI[np.where(ISI < cutoff)[0]]
		ISI_violations[i] = len(ISI_violation_instances) / len(ISI)

		unit_grp = preprocessing_file[session.subExperiment.name + '/' + session.name + '/group_0/' + str(unit)]
		unit_grp.create_dataset("ISI_violations", data=ISI_violations[i])

	return ISI_violations
