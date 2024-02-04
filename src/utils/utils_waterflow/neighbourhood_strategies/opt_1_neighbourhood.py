import numpy as np
from itertools import product
from copy import deepcopy

from src.utils.utils_waterflow.dow import DOW
from src.larp import LARP
from src.utils.gurobipy_utils import (fit, 
                                      add_constrs, 
                                      modify_rhs_constrs, 
                                      remove_constrs)


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
    dow.to_matrix()
    larp, constrs = add_constrs(larp, dow)
    dow.to_vector()
    
    neighbours = list()
    discarded_list = list()
    optimal_neighbours = list()

    for idx in range(len(dow.X)):
        tmp_X = deepcopy(dow.X)
        tmp_X[idx] = not dow.X[idx]
        dow_status_to_close = tmp_X[idx] == 0
        idx += 1 # adjust index of X to match the values in Y and Z

        if dow_status_to_close: # binary value is 0
            # change binary status to 1 (open)
            tmp = _change_status_to_close(larp, constrs, dow, tmp_X, idx)
        else: # binary value is 1
            # change binary status to 0 (close)
            tmp = _change_status_to_open(larp, constrs, dow, tmp_X, idx)
        
        good_neighbour, other_neighbours, discarded_dows = tmp

        optimal_neighbours.append(good_neighbour)
        neighbours.extend(other_neighbours)
        discarded_list.extend(discarded_dows)

    larp = remove_constrs(larp, constrs) # remove added constraints, no more needed

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

def _change_status_to_close(larp:LARP, constrs:dict, dow:DOW, tmp_X:np.ndarray, idx:int) -> tuple:
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

    cartesian = product(acceptable_values, repeat=len(indexes_positions_of_idx))

    for disp in cartesian:
        tmp_dow.Y[indexes_positions_of_idx] = disp
        tmp_Y = deepcopy(tmp_dow.Y)
        _feasibility_check(dow.m_storages, dow.n_fields, dow.k_vehicles, 
                          tmp_X, tmp_Y, tmp_dow.Z, larp, constrs, 
                          tmp_neighbours, discarded_dows)

    local_optimum, dows = _optimality_check(dow, tmp_neighbours)
    return local_optimum, dows, discarded_dows

def _change_status_to_open(larp:LARP, constrs:dict, dow:DOW, tmp_X:np.ndarray, idx:int) -> tuple:
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
    cartesian = product(uniques, repeat=len(dow.Y))

    for disp in cartesian:
        tmp_Y = np.array(disp)
        _feasibility_check(dow.m_storages, dow.n_fields, dow.k_vehicles, 
                          tmp_X, tmp_Y, tmp_Z, larp, constrs, 
                          tmp_neighbours, discarded_dows)

    local_optimum, dows = _optimality_check(dow, tmp_neighbours)
    return local_optimum, dows, discarded_dows

def _feasibility_check(m_storages:int, n_fields:int, k_vehicles:int, 
                       tmp_X:np.array, tmp_Y:np.array, tmp_Z:np.array, 
                       larp:LARP, constrs:dict, tmp_neighbours:list, discarded_dows:list) -> tuple:
    '''
    This is a suppot function used to create a new dow solution 
    and check if this solution is feasible.

    Arguments
    ---------
    m_storages:int
    Integer number of storages

    n_fields:int
    Integer number of fields

    k_vehicles:int
    Integer number of vehicles

    tmp_X:np.array
    Temporary X decision variable to create a new dow

    tmp_Y:np.array
    Temporary Y decision variable to create a new dow
    
    tmp_Z:np.array
    Temporary Z decision variable to create a new dow
    
    larp:LARP
    An instance of the LARP model
    
    constrs:dict
    Dictionary of additional constrains for LARP model
    
    tmp_neighbours:list
    List of temporary neighbours 
    
    discarded_dows:list
    List if discarded dows since they are not feasible
    
    Return
    ------
    None
    '''

    neighbour_dow = DOW(m_storages, n_fields, k_vehicles)
    neighbour_dow.X = tmp_X
    neighbour_dow.Y = tmp_Y
    neighbour_dow.Z = tmp_Z

    neighbour_dow.to_matrix()
    # print('neighbour dow:', neighbour_dow)

    # print('modify RHS constraints with discovered neighbour...')
    modify_rhs_constrs(neighbour_dow, constrs)
    # print('change completed')

    # print('fitting LARP model...')
    larp, is_fit = fit(larp, neighbour_dow)
    # print('fitting completed')

    neighbour_dow.to_vector()

    if is_fit:
        # print('neighbour dow is FEASIBLE')
        tmp_neighbours.append([neighbour_dow, neighbour_dow.obj_value])
    else:
        # print('neighbour dow is NOT FEASIBLE')
        discarded_dows.append(neighbour_dow)
    
def _optimality_check(dow:DOW, neighbours:list) -> tuple:
    '''
        This is a support function to determine the local optimum.

        Arguments
        ---------
        dow:DOW
        A drop-of-water (dow) rapresenting a certain solution of LARP model

        neighbours:list
        List of neighbours drop-of-waters (dows) for given dow

        Return
        ------
        local_optimum:DOW
        Drop-of-water (dow) rapresenting a certain local optimum
        
        dows:list
        List of non local optimum dows
    '''

    if neighbours: # i.e. current dow has a list of neighbour solutions
        # retrieve the best neighbour according the objective value
        dows, obj_vals = zip(*neighbours)
        idx_min = obj_vals.index(min(obj_vals))
        candidate = dows[idx_min]
    else:
        candidate = dow
        dows = []

    # compare current dow vs candidate dow for the role of local optimum
    if dow.obj_value <= candidate.obj_value:
        # print('local optimum is dow!')
        local_optimum = dow
        local_optimum.to_vector()
    else:
        # print('local optimum is candidate!')
        local_optimum = candidate
        dows = list(dows)

    return local_optimum, dows
