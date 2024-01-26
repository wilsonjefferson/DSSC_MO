import pickle
from itertools import product, dropwhile
import os

from src.get_data import random_data
from src.larp import LARP
# from src.plot_scalability import plot_scalability

 
if __name__ == '__main__':

    backup = os.getcwd() + '\\DSSC_MO\\backup\\scalability.pkl'

    k_vehicles = 5
    Q_vehicle_capacity = 2000
    facility = 'F'

    n_iterations = 10
    iterations = range(n_iterations)
    n_fields_instances = [100, 500]
    m_storages_instances = [10, 20, 30, 40, 50, 100]

    cartesian = product(iterations, n_fields_instances, m_storages_instances)
    if os.path.exists(backup):
        with open(backup, 'rb') as f:
            scalability = pickle.load(f)
        computed = [(i,j,k) for i, j, k, _ in scalability]
    else:
        scalability, computed = list(), list()

    for itr, n_f, m_s in dropwhile(lambda x: x in computed, cartesian):
        print('itr:', itr, 'n_f:', n_f, 'm_s:', m_s)

        for iter in iterations[itr:]:
            params = random_data(n_f, m_s)
            fields, storages, households = params[0], params[1], params[2]
            f, q = params[3], params[4]
            fs_dist, cs_dist, d = params[5], params[6], params[7]

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
                            d)
        
            larp_model.build()
            larp_model.optimize()
        
            # NOTE: scalability = [iter, num. of storages, num. of fields, runtime (sec)]
            scalability.append([iter, m_s, n_f, larp_model.get_execution_time()])
            with open(backup, 'wb') as f:
                pickle.dump(scalability, f)
        
            larp_model.dispose()
    
    # folder_path = os.getcwd() + '\\DSSC_MO\\images\\'
    # save_at = folder_path + 'scalability.svg'
    # plot_scalability(scalability, save_at)
