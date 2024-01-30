import numpy as np
from itertools import product
from copy import deepcopy

from src.utils.utils_waterflow.dow import DOW
from src.larp import LARP
from src.utils.gurobipy_utils import (fit, 
                                      add_constrs, 
                                      modify_rhs_constrs, 
                                      remove_constrs)


def swap(larp:LARP, dow:DOW) -> tuple:

    dow.to_matrix()

    neighbours = list()
    discarded_dows = list()
    constrs = None

    X_idx_zeros = np.nonzero(dow.X == 0)[0]
    X_idx_nonzeros = np.nonzero(dow.X)[0]

    # print('X_idx_zeros:', X_idx_zeros)
    # print('X_idx_nonzeros:', X_idx_nonzeros)
    
    cartesian = product(X_idx_zeros, X_idx_nonzeros)
    for zero_idx, nonzero_idx in cartesian:
        # print('zero_idx:', zero_idx, 'nonzero_idx:', nonzero_idx)

        tmp_X = deepcopy(dow.X)
        tmp_X[zero_idx] = 1
        tmp_X[nonzero_idx] = 0
                   
        tmp_Y = deepcopy(dow.Y)
        nonzero_rows = np.nonzero(dow.Y[:,nonzero_idx])[0]
        
        # print('Y nonzero_rows:', nonzero_rows)
        tmp_Y[nonzero_rows, zero_idx] = 1
        tmp_Y[nonzero_rows, nonzero_idx] = 0

        tmp_Z = deepcopy(dow.Z)
        nonzero_row = np.nonzero(dow.Z[:,nonzero_idx])[0]
        # print('Z nonzero_row:', nonzero_row)
        tmp_Z[nonzero_row, zero_idx] = 1
        tmp_Z[nonzero_row, nonzero_idx] = 0

        nonzero_column = np.nonzero(dow.Z[nonzero_idx,:])[0]
        # print('Z nonzero_column:', nonzero_column)
        tmp_Z[zero_idx, nonzero_column] = 1
        tmp_Z[nonzero_idx, nonzero_column] = 0

        neighbour_dow = DOW(dow.m_storages, dow.n_fields, dow.k_vehicles)
        neighbour_dow.X = tmp_X
        neighbour_dow.Y = tmp_Y
        neighbour_dow.Z = tmp_Z

        # print('neighbour dow:', neighbour_dow)

        if constrs:
            # print('modify existing constraints')
            modify_rhs_constrs(neighbour_dow, constrs)
        else:
            # print('add new constraints')
            larp, constrs = add_constrs(larp, neighbour_dow)

        larp, is_fit = fit(larp, neighbour_dow)
        neighbour_dow.to_vector()
        
        if is_fit:
            # print('neighbour dow is FEASIBLE')
            neighbours.append([neighbour_dow, neighbour_dow.obj_value])
        else:
            # print('neighbour dow is NOT FEASIBLE')
            discarded_dows.append(neighbour_dow)

    # print('neighbours:', neighbours)
    larp = remove_constrs(larp, constrs)
    
    if neighbours:
        dows, obj_vals = zip(*neighbours)
        obj_vals = [dow.obj_value for dow in dows]
        idx_min = obj_vals.index(min(obj_vals))
        candidate = dows[idx_min]
    else:
        candidate = dow
        dows = []

    if dow.obj_value <= candidate.obj_value:
        # print('local optimum is dow!')
        local_optimum = dow
        local_optimum.to_vector()
    else:
        # print('local optimum is candidate!')
        local_optimum = candidate
        dows = list(dows)
        dows.remove(candidate)
    
    return local_optimum, dows, discarded_dows
