import numpy as np
from utils.experiment_classes import *

class thirty_channel_ecog(Probe):
    def __init__(self):
        self.nr_of_groups = 1
        self.type = 'array'
        self.nr_of_electrodes_per_group = 30
        self.nr_of_electrodes = 30

    def get_channel_mapping(self, amplifier):
        if amplifier == 'rhd':
            idx = {0:[4,3,1,30,28,27,6,5,2,29,26,25,8,9,7,24,22,23,10,12,14,17,19,21,11,13,15,16,18,20]}
        elif amplifier == 'rhs':
            idx = {0:[28,27,25,22,20,19,30,29,26,21,18,17,0,1,31,16,14,15,2,4,6,9,11,13,3,5,7,8,10,12]}
        self.id = idx

    def get_channel_coords(self):
        self.coords = [[0,0],[50,0],[100,0],[150,0],[200,0],[250,0],[0,50],[50,50],[100,50],[150,50],[200,50],[250,50],[0,100],[50,100],[100,100],[150,100],[200,100],[250,100],[0,150],[50,150],[100,150],[150,150],[200,150],[250,150],[0,200],[50,200],[100,200],[150,200],[200,200],[250,200]]
