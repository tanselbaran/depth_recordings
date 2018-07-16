import numpy as np
probe_info = {}
probe_info['nr_of_groups'] = 4
probe_info['type'] = 'array'
probe_info['nr_of_electrodes_per_group'] = 8
probe_info['numTrodes'] = 32

###Channel mapping###
id = {}
electrode = 0
for group in range(probe_info['nr_of_groups']):
    id[group] = []
    for channel in range(probe_info['nr_of_electrodes_per_group']):
        id[group].append(electrode)
        electrode = electrode + 1
probe_info['id'] = id
