from dataclasses import dataclass

import pandas as pd
import numpy as np
import random
from copy import deepcopy

SEED = 66
rng = np.random.default_rng(SEED)


@dataclass(frozen=False)
class DOW:

    def __init__(self, m_storages:int, n_fields:int, k_vehicles:int):

        self.m_storages = m_storages
        self.n_fields = n_fields
        self.k_vehicles = k_vehicles

        self.l_storages, index_storages = self._rand_storages()
        if any(index_storages):
            self.l_fields = self._rand_fields(index_storages)
            self.l_vehicles = self._rand_vehicles(index_storages)
        else:
            self.l_fields = np.ones((n_fields,), dtype=int)
            self.l_vehicles = np.ones((k_vehicles,), dtype=int)

        self.X = self.l_storages

        Y_coordinates = np.array([[i, j-1] for i, j in enumerate(self.l_fields)])
        rows, cols = zip(*Y_coordinates)
        self.Y = np.zeros((n_fields, m_storages), dtype=int)
        self.Y[rows, cols] = 1

        self.Z = np.zeros((m_storages+1, m_storages+1), dtype=int)

        for u_idx in range(len(self.l_vehicles)):
            u = self.l_vehicles[u_idx]
            u = u if u == 0 else u-1
            v_idx = u_idx+1
            v = self.l_vehicles[v_idx] if v_idx < len(self.l_vehicles) else 0
            v = v if v == 0 else v-1
            self.Z[u,v] = 1

        self.obj_value = None

    def __str__(self):
        out_string = f'objValue: {self.obj_value}\n\n'
        out_string += f'X: {self.X}\n\n'
        out_string += f'Y: {self.Y}\n\n'
        out_string += f'Z: {self.Z}\n\n'
        return out_string

    def __eq__(self, other):
        if isinstance(other, DOW):
            return self.X == other.X and \
                    self.Y == other.Y \
                    and self.Z == other.Z
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def _rand_storages(self):
        l_storages = rng.integers(2, size=self.m_storages)
        index_storages = [i+1 for i in np.nonzero(l_storages)[0]]
        return l_storages, index_storages
    
    def _rand_fields(self, index_storages:list):
        l_fields = random.choices(index_storages, k=self.n_fields)
        return l_fields
    
    def _rand_vehicles(self, index_storages:list):
        l_vehicles = list()
        n_available_storages = deepcopy(index_storages)
        for f in range(self.k_vehicles):
            l_vehicles.append(0)
            
            if f+1 < self.k_vehicles:
                n_visits = rng.integers(len(n_available_storages), size=1)[0]
            else:
                n_visits = len(n_available_storages)
            
            random_storages = random.sample(n_available_storages, k=n_visits)
            l_vehicles.extend(random_storages)

            # Pop the selected items from the list
            for storage in random_storages:
                n_available_storages.remove(storage)

        return l_vehicles
            