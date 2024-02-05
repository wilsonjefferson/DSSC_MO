import numpy as np
import functools as ft
import multiprocessing as mp

from itertools import product
from copy import deepcopy

from src.utils.utils_waterflow.dow import DOW
from src.larp import LARP
from src.utils.gurobipy_utils import create_larp, add_constrs, remove_constrs
from src.utils.utils_waterflow.neighbourhood_strategies.support_functions import feasibility_check, optimality_check


N_PROCESSES = mp.cpu_count()


def opt_1(larp:LARP, dow:DOW) -> tuple:
    '''
        Opt1 is a neighbourhood structure used during the local search
        algorithm to identify the list of valid neighbours of a certain
        solution, also called drop-of-water (dow).

        Given dow.X (binary list), change a 1 value with a 0 value and repeat
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

    # print('opt_1 | dow vector:', dow)
    
    neighbours = list()
    discarded_list = list()
    optimal_neighbours = list()

    for idx in range(len(dow.X)):
        tmp_X = deepcopy(dow.X)
        tmp_X[idx] = not dow.X[idx]
        dow_new_status_to_close = tmp_X[idx] == 0
        idx += 1 # adjust index of X to match the values in Y and Z

        if dow_new_status_to_close: # binary value is 0
            # change binary status to 1 (open)
            # tmp = _change_status_to_close(larp, constrs, dow, tmp_X, idx)
            tmp = _change_status_to_close(larp, dow, tmp_X, idx)

        else: # binary value is 1
            # change binary status to 0 (close)
            # tmp = _change_status_to_close(larp, constrs, dow, tmp_X, idx)
            tmp = _change_status_to_open(larp, dow, tmp_X, idx)
        
        good_neighbour, other_neighbours, discarded_dows = tmp

        optimal_neighbours.append(good_neighbour)
        neighbours.extend(other_neighbours)
        discarded_list.extend(discarded_dows)

    if optimal_neighbours: # i.e. current optimal has a list of neighbour solutions
        # retrieve the best neighbour according the objective value
        obj_vals = [neighbour.obj_value for neighbour in optimal_neighbours]
        idx_min = obj_vals.index(min(obj_vals))
        candidate = optimal_neighbours[idx_min]
        dows = neighbours
    else:
        candidate = dow
        dows = []

    # compare current dow vs candidate dow for the role of local optimum
    if dow.obj_value <= candidate.obj_value:
        # print('local optimum is dow!')
        local_optimum = dow
        dows.extend(optimal_neighbours)
    else:
        # print('local optimum is candidate!')
        local_optimum = candidate
        optimal_neighbours.remove(candidate)
        neighbours.extend(optimal_neighbours)
        dows = neighbours
    
    return local_optimum, dows, discarded_list

def _change_status_to_close(larp:LARP, dow:DOW, tmp_X:np.ndarray, idx:int) -> tuple:
    '''
        If change status from 1 (open) to 0 (close), following routine
        is executed to adjust Y and Z attributes and generate new dows.
        These dows are evaluated against the LARP model to determine which
        dows are feasible. 

        Arguments
        ---------
        larp:LARP
        An instance of the LARP model

        constrs:dict
        Dictionary of additional constrains for LARP model

        dow:DOW
        A drop-of-water (dow) representing a certain solution

        tmp_X:np.ndarray
        A copy of dow.X array

        idx:int
        Integer number representing the position of the changed binary value

        Return
        ------
        local_optimum:DOW
        New local optimum solution

        dows:list
        List of neighbours dows of the local optimum

        discarded_dows:list
        List of no feasible solutions
    '''

    discarded_dows = list()
    tmp_neighbours = list()
    
    # a no-really elegant way to adjust a copy of dow.Z
    # to match the changes from dow.X 
    tmp_dow = deepcopy(dow)

    # adapt decision variable Z
    unwanted_value_idx = np.nonzero(tmp_dow.Z == idx)[0]
    if unwanted_value_idx+1 < len(tmp_dow.Z):
        if tmp_dow.Z[unwanted_value_idx-1] == tmp_dow.Z[unwanted_value_idx+1]:
            tmp_dow.Z = np.delete(tmp_dow.Z, unwanted_value_idx+1)
    np.delete(tmp_dow.Z, unwanted_value_idx)
    if tmp_dow.Z[-1] == 0:
        tmp_dow.Z = np.delete(tmp_dow.Z, -1)

    # adapt decision variable Y
    indexes_positions_of_idx = np.nonzero(dow.Y == idx)
    uniques = np.unique(dow.Y)
    acceptable_values = np.nonzero(uniques != idx)
    acceptable_values = uniques[acceptable_values]

    cartesian = list(product(acceptable_values, repeat=len(indexes_positions_of_idx)))

    chunks = [cartesian[x:x+N_PROCESSES] for x in range(0, len(cartesian), N_PROCESSES)]
    feasibility_params = [tmp_X, tmp_dow.Z, tmp_neighbours, discarded_dows]

    process_task = ft.partial(task_to_close, inputs=larp.inputs, dow=dow, tmp_dow_Y=tmp_dow.Y, 
                                indexes_positions_of_idx=indexes_positions_of_idx, 
                                params=feasibility_params)

    print('start parallelization...')
    with mp.Pool(N_PROCESSES) as pool:
        pool.map(process_task, chunks)
    print('parallelization completed.')

    local_optimum, dows = optimality_check(dow, tmp_neighbours)
    return local_optimum, dows, discarded_dows

def task_to_close(chunk:list, inputs:dict, dow:DOW, tmp_dow_Y:np.array, indexes_positions_of_idx:np.array, params:list):
    larp = create_larp(inputs)

    dow.to_matrix()
    larp, constrs = add_constrs(larp, dow)
    dow.to_vector()

    feasibility_params = deepcopy(params)
    feasibility_params.insert(2, larp)
    feasibility_params.insert(3, constrs)
    feasibility_params = [dow.m_storages, dow.n_fields, dow.k_vehicles]+feasibility_params
    feasibility_params.insert(4, None)

    print('close | feasibility_params:', len(feasibility_params))

    for disp in chunk:
        tmp_dow_Y[indexes_positions_of_idx] = disp
        tmp_Y = deepcopy(tmp_dow_Y)
        feasibility_params[4] = tmp_Y
        feasibility_check(*feasibility_params)
    
    print('neighbours:', len(feasibility_params[-2]))
    print('discarded:', len(feasibility_params[-1]))
    
    larp = remove_constrs(larp, constrs) # remove added constraints, no more needed

def _change_status_to_open(larp:LARP, dow:DOW, tmp_X:np.ndarray, idx:int) -> tuple:
    '''
        If change status from 0 (close) to 0 (open), following routine
        is executed to adjust Y and Z attributes and generate new dows.
        These dows are evaluated against the LARP model to determine which
        dows are feasible. 

        Arguments
        ---------
        larp:LARP
        An instance of the LARP model

        constrs:dict
        Dictionary of additional constrains for LARP model

        dow:DOW
        A drop-of-water (dow) representing a certain solution

        tmp_X:np.ndarray
        A copy of dow.X array

        idx:int
        Integer number representing the position of the changed binary value

        Return
        ------
        local_optimum:DOW
        New local optimum solution

        dows:list
        List of neighbours dows of the local optimum

        discarded_dows:list
        List of no feasible solutions
    '''

    discarded_dows = list()
    tmp_neighbours = list()

    # adapt decision variable Z
    tmp_Z = deepcopy(dow.Z)

    if len(tmp_Z) == 0 or tmp_Z is None:
        tmp_Z = np.array([0, idx])
    else:
        tmp_Z = np.append(tmp_Z, idx)

    # adapt decision variable Y
    uniques = np.unique(dow.Y)
    uniques = np.append(uniques, idx)
    cartesian = list(product(uniques, repeat=len(dow.Y)))
    
    chunks = [cartesian[x:x+N_PROCESSES] for x in range(0, len(cartesian), N_PROCESSES)]
    feasibility_params = [tmp_X, tmp_Z, tmp_neighbours, discarded_dows]

    process_task = ft.partial(task_to_open, inputs=larp.inputs, dow=dow,
                                params=feasibility_params)

    print('start parallelization...')
    with mp.Pool(N_PROCESSES) as pool:
        pool.map(process_task, chunks)
    print('parallelization completed.')

    local_optimum, dows = optimality_check(dow, tmp_neighbours)
    return local_optimum, dows, discarded_dows

def task_to_open(chunk:list, inputs:dict, dow:DOW, params:list):
    larp = create_larp(inputs)

    dow.to_matrix()
    larp, constrs = add_constrs(larp, dow)
    dow.to_vector()

    feasibility_params = deepcopy(params)
    feasibility_params.insert(2, larp)
    feasibility_params.insert(3, constrs)
    feasibility_params = [dow.m_storages, dow.n_fields, dow.k_vehicles]+feasibility_params
    feasibility_params.insert(4, None)

    print('open | feasibility_params:', len(feasibility_params))

    for disp in chunk:
        tmp_Y = np.array(disp)
        feasibility_params[4] = tmp_Y
        feasibility_check(*feasibility_params)

    print('tmp_neighbours:', len(feasibility_params[-2]))
    print('discarded:', len(feasibility_params[-1]))

    larp = remove_constrs(larp, constrs) # remove added constraints, no more needed
