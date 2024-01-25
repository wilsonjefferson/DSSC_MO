import numpy as np
import pickle
from itertools import product, dropwhile
import os

from src.get_data import random_data
from src.larp import LARP
from src.plot_scalability import plot_scalability


if __name__ == '__main__':

    folder_path = os.getcwd() + '\\DSSC_MO\\backup\\backup_scalability\\'
    backup = folder_path + 'scalability.pkl'

    k_vehicles = 5
    Q_vehicle_capacity = 2000
    facility = 'F'

    n_fields_instances = [100, 500, 1000]
    m_storages_instances = [10, 20, 50, 100, 200]

    use = (max(n_fields_instances), max(m_storages_instances))
    _fields, _storages, _households, _f, _q, _fs_dist, _cs_dist, _pivot_d = random_data(*use)

    cartesian = product(n_fields_instances, m_storages_instances)

    if os.path.exists(backup):
        with open(backup, 'rb') as f:
            scalability = pickle.load(f)
        computed = [(i,j) for i, j, _ in scalability]
    else:
        scalability = list()
        computed = list()

    for n_f, m_s in dropwhile(lambda x: x in computed, cartesian):
    
        print('n_f:', n_f, 'm_s:', m_s)

        fields = _fields[:n_f]
        storages = _storages[:m_s]
        households = _households[:n_f]

        f = {k:v for k, v in _f.items()}
        q = {k:v for k, v in _q.items()}
        
        cs_dist = _cs_dist.iloc[:n_f, :m_s]
        pivot_d = _pivot_d.iloc[:n_f, :n_f]
        fs_dist = _fs_dist.iloc[np.r_[:m_s, len(_fs_dist.columns)-1], 
                                np.r_[:m_s, len(_fs_dist.columns)-1]]

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
        
        # NOTE: scalability = [num. of storages, num. of fields, runtime (sec)]
        scalability.append([m_s, n_f, larp_model.get_execution_time()])
        with open(backup, 'wb') as f:
            pickle.dump(scalability, f)
        
        larp_model.dispose()
    
    folder_path = os.getcwd() + '\\DSSC_MO\\images\\'
    save_at = folder_path + 'scalability.svg'
    plot_scalability(scalability, save_at)
