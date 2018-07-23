import numpy as np
from utils.experiment_classes import *
class thirtytwo_channel_ecog(Probe):
    def __init__(self,):
        self.nr_of_groups = 4
        self.type = 'array'
        self.nr_of_electrodes_per_group = 8
        self.nr_of_electrodes = 32

    def get_channel_mapping(self, amplifier):
        if amplifier == 'rhd':
            id = {
            0:[0,1,2,3,4,5,6,7],
            1:[8,9,10,11,12,13,14,15],
            2:[16,17,18,19,20,21,22,23],
            3:[24,25,26,27,28,29,30,31]
            }
        elif amplifier == 'rhs':
            id = {
            0:[24,25,26,27,28,29,30,31],
            1:[0,1,2,3,4,5,6,7],
            2:[8,9,10,11,12,13,14,15],
            3:[16,17,18,19,20,21,22,23]
            }
        self.id = id

    def get_channel_coords(self):
        self.coords = [[[80,80],[120,40],[80,0],[0,0],[40,40],[0,80],[40,120],[120,120]],
        [[80,0],[0,0],[40,40],[0,80],[40,120],[80,80],[120,120],[120,40]],
        [[0,40],[0,120],[40,80],[80,120],[120,80],[80,40],[120,0],[40,0]],
        [[0,120],[80,120],[120,80],[80,40],[120,0],[40,0],[0,40],[40,80]]]
