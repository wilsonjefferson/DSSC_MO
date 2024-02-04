import numpy as np

from src.utils.utils_waterflow.dow import DOW
from src.larp import LARP
from src.utils.gurobipy_utils import (fit, 
                                      add_constrs, 
                                      modify_rhs_constrs)

def feasibility_check(m_storages:int, n_fields:int, k_vehicles:int, 
                       tmp_X:np.array, tmp_Y:np.array, tmp_Z:np.array, 
                       larp:LARP, constrs:dict, tmp_neighbours:list, discarded_dows:list) -> None:
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

    if len(constrs) != 0:
        # print('modify RHS constraints with discovered neighbour...')
        modify_rhs_constrs(neighbour_dow, constrs)
    else:
        # print('add new constraints')
        larp, constrs = add_constrs(larp, neighbour_dow)

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
    
def optimality_check(dow:DOW, neighbours:list) -> tuple:
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
