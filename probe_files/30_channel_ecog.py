import numpy as np
probe_info = {}
probe_info['nr_of_groups'] = 1
probe_info['type'] = 'array'
probe_info['nr_of_electrodes_per_group'] = 1
probe_info['numTrodes'] = 30

###Channel mapping###
id = {}
electrode = 0
for group in range(probe_info['nr_of_groups']):
    id[group] = []
    for channel in range(probe_info['nr_of_electrodes_per_group']):
        id[group].append(electrode)
        electrode = electrode + 1
probe_info['id'] = id

coords = [[100,0], [100,50],[50,0],[0,0],[50,50],[0,50],[100,100],[0,100],[50,100],[0,150],[0,200],[50,150],[50,200],[100,150],[100,200],[150,200],[150,150],[200,200],[200,150],[250,200],[250,150],[200,100],[250,150],[150,100],[250,50],[200,50],[250,0],[200,0],[150,50],[150,0]]
probe_info['coords'] = coords
