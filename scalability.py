import pandas as pd
import numpy as np
import os
import itertools

from src.get_data import get_data
from src.larp import LARP
from src.plot_scalability_trend import plot_scalability


if __name__ == '__main__':

    n_fields_instances = [100, 500, 1000]
    m_storages_instances = [10, 20, 50, 100, 200]
    use = (max(n_fields_instances), max(m_storages_instances))

    facility, k_vehicles, Q_vehicle_capacity, _fields, \
        _storages, _households, _f, _q, _fs_dist, _cs_dist, _pivot_d = get_data(use)
    
    field_storage_slices = list(itertools.product(n_fields_instances, m_storages_instances))
    scalability = list()

    for n_f, m_s in field_storage_slices:
        fields = _fields[:n_f]
        storages = _storages[:m_s]
        households = _households[:n_f]

        f = {k:v for k, v in _f.items()}
        q = {k:v for k, v in _q.items()}
        
        cs_dist = _cs_dist.iloc[:n_f, :m_s]
        fs_dist = _fs_dist.iloc[np.r_[:m_s+1, len(_storages)], np.r_[:m_s+1, len(_storages)]]
        pivot_d = _pivot_d.iloc[:n_f, :n_f]

        larp_model = LARP(facility, 
                        k_vehicles, 
                        Q_vehicle_capacity, 
                        fields, 
                        storages, 
                        households, 
                        f, 
                        q, 
                        fs_dist, 
                        cs_dist, 
                        pivot_d)
        
        larp_model.optimize()
        scalability.append([n_f, m_s, larp_model.get_objvalues()['runtime']])
        larp_model.dispose()
    
    plot_scalability(scalability)
