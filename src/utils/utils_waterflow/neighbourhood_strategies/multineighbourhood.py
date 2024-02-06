
import numpy as np
from copy import deepcopy

from src.utils.utils_waterflow.dow import DOW

from src.utils.gurobipy_utils import create_larp, add_constrs
from src.utils.utils_waterflow.neighbourhood_strategies.support_functions import feasibility_check


def process_initializer(larp_inputs:dict, dow:DOW, params:list):

    global tmp_neighbours
    global discarded_dows
    global feasibility_params

    tmp_neighbours = list()
    discarded_dows = list()

    larp = create_larp(larp_inputs)
    dow.to_matrix()
    larp, constrs = add_constrs(larp, dow)
    dow.to_vector()

    feasibility_params = deepcopy(params)
    feasibility_params.insert(4, None)
    feasibility_params.insert(6, larp)
    feasibility_params.insert(7, constrs)
    
def task_to_close(disp:tuple, tmp_dow_Y:np.array, indexes_positions_of_idx:np.array):

    global tmp_neighbours
    global discarded_dows
    global feasibility_params

    tmp_dow_Y[indexes_positions_of_idx] = disp
    tmp_Y = deepcopy(tmp_dow_Y)
    feasibility_params[4] = tmp_Y
    return feasibility_check(*feasibility_params)

def task_to_open(disp:tuple):

    global tmp_neighbours
    global discarded_dows
    global feasibility_params

    tmp_Y = np.array(disp)
    feasibility_params[4] = tmp_Y
    return feasibility_check(*feasibility_params)

def task_to_swap(dow:DOW, indexes:tuple):

    global tmp_neighbours
    global discarded_dows
    global feasibility_params

    zero_idx, nonzero_idx = indexes

    zero_idx += 1
    nonzero_idx += 1
    # print('zero_idx:', zero_idx, 'nonzero_idx:', nonzero_idx)

    # adjust X decision variable
    tmp_X = deepcopy(dow.X)
    tmp_X[zero_idx-1] = 1
    tmp_X[nonzero_idx-1] = 0

    # adjust Y decision variable
    tmp_Y = deepcopy(dow.Y)
    reassign_indexes = np.nonzero(tmp_Y == nonzero_idx)[0]
    tmp_Y[reassign_indexes] = zero_idx
    # print('tmp_Y:', tmp_Y)

    # adjust Z decision variable
    tmp_Z = deepcopy(dow.Z)
    reassign_indexes = np.nonzero(tmp_Z == nonzero_idx)[0]
    tmp_Z[reassign_indexes] = zero_idx
    # print('tmp_Z:', tmp_Z)

    feasibility_params[3] = tmp_X
    feasibility_params[4] = tmp_Y
    feasibility_params[5] = tmp_Z

    return feasibility_check(*feasibility_params)
