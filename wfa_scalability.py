from timeit import default_timer as timer
from itertools import product, dropwhile
import os
import pickle

from src.utils.get_data import random_data
from src.larp import LARP
from src.waterflow import waterflow

 
if __name__ == '__main__':

    # location where store the scalability data 
    backup = os.getcwd() + '\\DSSC_MO\\backup\\wfa_scalability.pkl'
    print('backup:', backup)

    # LARP model paramenters
    n_iterations = 10
    iterations = range(n_iterations)
    n_fields_instances = [5]
    m_storages_instances = [5, 6, 7, 8]
    k_vehicles_instances = [2, 3, 4, 5]

    Q_vehicle_capacity = 100
    facility = 'F'

    # WFA parameters
    max_cloud = 3
    max_pop = 10

    min_ero = 2
    max_UIE = 5

    best_solution = None

    # cartesian product among the number of iteration, number of
    # fields and number of storages
    cartesian = product(n_fields_instances, m_storages_instances, iterations)

    # check if a backup of the scalability already exist
    # this is in case the program was interrupted before its complention
    if os.path.exists(backup):
        with open(backup, 'rb') as file:
            scalability = pickle.load(file)
        
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
        
        # k_v = k_vehicles_instances[0]

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

            larp = LARP(facility, 
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
            
            # NOTE: Since it is difficult to find an optimal solution 
            # trying with random X, Y and Z (decision variables)
            # we fix a random X and set guroby to find and stop the very first feasible solution
            larp.model.setParam('OutputFlag', 0)
            larp.model.setParam('SolutionLimit', 1)

            print('model build in-progress...')
            start_build = timer()
            larp.build()
            end_build = timer()
            build_time = end_build-start_build
            print('Build time:', build_time)

            print('model optimization in-progress...')
            start_opt = timer()
            best_solution = waterflow(larp, max_cloud, max_pop, max_UIE, min_ero)
            end_opt = timer()
            opt_time = end_opt-start_opt
            print('Optimization time:', opt_time)

            if best_solution is None:
                print('problem is INFEASIBLE')
            else:
                # NOTE: scalability = [num. of fields, num. of storages, num. of k_vehicles, iter, build_time (sec), opt_time (sec)]
                scalability.append([n_f, m_s, k_v, itr, build_time, opt_time])

                # save updated information in scalability
                with open(backup, 'wb') as file:
                    pickle.dump(scalability, file)
            
                larp.dispose()
                break
