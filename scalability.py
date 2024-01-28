import pickle
from timeit import default_timer as timer
from itertools import product, dropwhile
import os
from gurobipy import GRB

from src.get_data import random_data
from src.larp import LARP
# from src.plot_scalability import plot_scalability

 
if __name__ == '__main__':

    # location where store the scalability data 
    backup = os.getcwd() + '\\DSSC_MO\\backup\\scalability2.pkl'
    print('backup:', backup)

    # LARP model paramenters
    n_iterations = 10
    iterations = range(n_iterations)
    n_fields_instances = [100, 300]
    m_storages_instances = [10, 20, 30, 50, 70, 100]
    k_vehicles_instances = [3, 6, 9, 15, 21, 30]

    Q_vehicle_capacity = 2000
    facility = 'F'

    # cartesian product among the number of iteration, number of
    # fields and number of storages
    cartesian = product(n_fields_instances, m_storages_instances, iterations)

    # check if a backup of the scalability already exist
    # this is in case the program was interrupted before its complention
    if os.path.exists(backup):
        with open(backup, 'rb') as f:
            scalability = pickle.load(f)
        
        # list of cases already evaluated
        computed = [(i,j,k) for i, j, _, k, _, _ in scalability]
    else:
        # initially no cases is evaluated
        scalability, computed = list(), list()

    # NOTE: dropwhile filter those points of the cartesian variable
    # which are already evaluated
    for n_f, m_s, itr in dropwhile(lambda x: x in computed, cartesian):
        k_v = m_storages_instances.index(m_s)
        k_v = k_vehicles_instances[k_v]

        while True:
            print('n_f:', n_f, 'm_s:', m_s, 'k_v:', k_v, 'instance:', itr)
            
            # generate ranodm data for a given size of the problem
            print('generate random data in-progress...')
            start_data = timer()
            params = random_data(n_f, m_s)
            end_data = timer()
            print('Data generation time:', end_data-start_data)

            fields, storages, households = params[0], params[1], params[2]
            f, q = params[3], params[4]
            fs_dist, cs_dist, d = params[5], params[6], params[7]
            k_vehicles = k_v

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

            print('model build in-progress...')
            start_build = timer()
            larp_model.build()
            end_build = timer()
            build_time = end_build-start_build
            print('Build time:', build_time)

            print('model optimization in-progress...')
            start_opt = timer()
            larp_model.optimize()
            end_opt = timer()
            print('Optimization time:', end_opt-start_opt)

            print('model status:', larp_model.model.status)
            if larp_model.model.status in [GRB.INFEASIBLE, GRB.INF_OR_UNBD, GRB.UNBOUNDED]:
                print('problem is INFEASIBLE')
            else:
                # NOTE: scalability = [num. of fields, num. of storages, num. of k_vehicles, iter, runtime (sec)]
                scalability.append([n_f, m_s, k_v, itr, build_time, larp_model.get_execution_time()])

                # save updated information in scalability
                with open(backup, 'wb') as f:
                    pickle.dump(scalability, f)
            
                larp_model.dispose()
                break
    
    # folder_path = os.getcwd() + '\\DSSC_MO\\images\\'
    # save_at = folder_path + 'scalability.svg'
    # plot_scalability(scalability, save_at)
