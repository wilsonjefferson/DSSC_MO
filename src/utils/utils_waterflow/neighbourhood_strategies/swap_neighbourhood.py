import numpy as np
import functools as ft
import multiprocessing as mp
from itertools import product

from src.utils.utils_waterflow.dow import DOW
from src.larp import LARP
from src.utils.utils_waterflow.neighbourhood_strategies.multineighbourhood import process_initializer, task_to_swap
from src.utils.utils_waterflow.neighbourhood_strategies.support_functions import optimality_check


N_PROCESSES = mp.cpu_count()


def swap(larp:LARP, dow:DOW) -> tuple:
    '''
        Swap is a neighbourhood structure used during the local search
        algorithm to identify the list of valid neighbours of a certain
        solution, also called drop-of-water (dow).

        Given dow.X (binary list), swap a 1 value with a 0 value and repeat
        it for any 1s and 0s in dow.X, than adjust dow.Y and dow.Z and evaluate
        the new dow with the LARP model, i.e. check if the neighbour dow is feasible.

        The neighbourhood structure return the improved solution if exist.

        Arguments
        ---------
        larp:LARP
        An instance of the LARP model

        dow:DOW
        A drop-of-water (dow) representing a certain solution

        Return
        ------
        local_optimum:DOW
        An improved solution

        dows:list
        List of neighbours drop-of-waters (dows) for local optimum

        discarded_dows:list
        List of no feasible solutions
    '''

    neighbours = list()
    discarded_dows = list()

    X_idx_zeros = np.nonzero(dow.X == 0)[0]
    X_idx_nonzeros = np.nonzero(dow.X)[0]

    if len(X_idx_zeros) == 0 or len(X_idx_nonzeros) == 0:
        return dow, list(), discarded_dows
    
    cartesian = product(X_idx_zeros, X_idx_nonzeros)
        
    feasibility_params = [dow.m_storages, dow.n_fields, dow.k_vehicles, None, None]
    process_task = ft.partial(task_to_swap, dow)

    # print('start parallelization...')
    with mp.Pool(N_PROCESSES, process_initializer, (larp.inputs, dow, feasibility_params)) as pool:
        for results in pool.imap_unordered(process_task, cartesian, N_PROCESSES):
            neighbours.extend(results[0])
            discarded_dows.extend(results[1])
    # print('parallelization completed.')
    
    local_optimum, dows = optimality_check(dow, neighbours)
    return local_optimum, dows, discarded_dows
