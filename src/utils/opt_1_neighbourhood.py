import numpy as np
from sympy.utilities.iterables import multiset_permutations
from itertools import product
from copy import deepcopy

from DSSC_MO.src.utils.dow import DOW
from src.larp import LARP
from src.utils.gurobipy_utils import (fit, 
                                      add_constrs, 
                                      modify_rhs_constrs, 
                                      remove_constrs)


def opt_1(larp:LARP, dow:DOW) -> list:
    
    larp, constrs = add_constrs(larp, dow)

    neighbours = list()
    for idx in range(len(dow.X)):
        print('idx:', idx)
        tmp_X = deepcopy(dow.X)
        tmp_X[idx] = not dow.X[idx]
        if tmp_X[idx] == 0:
            larp, adj = _change_status_to_close(larp, constrs, dow, tmp_X, idx)
            neighbours.append(adj)
        else:
            larp, adj = _change_status_to_open(larp, constrs, dow, tmp_X, idx)
            neighbours.append(adj)

    larp = remove_constrs(larp, constrs)
    
    _, dows, discarded_dows = zip(*neighbours)
    dows = dows[dows != np.array(None)]
    obj_vals = [dow.obj_value for dow in dows]
    idx_min_obj_val = obj_vals.index(min(obj_vals))
    candidate = dows[idx_min_obj_val]

    local_optimum = dow if dow.obj_value <= candidate.obj_value else candidate
    return local_optimum

def _change_status_to_close(larp, constrs, dow, tmp_X, idx) -> tuple:

    # ----------------------------------

    tmp_Z = deepcopy(dow.Z)
    tmp_Z[idx,:] = 0
    tmp_Z[:,idx] = 0

    # ----------------------------------

    columns_nozero = [dow.Y[:, j].any() if j != idx else False for j in range(dow.X.shape[0])]
    colms_idx =  [j for j in range(len(columns_nozero)) if columns_nozero[j]]
    print('colms_idx:', colms_idx)

    n_cols = len(colms_idx)
    print('n_cols:', n_cols)

    column = dow.Y[:, idx]
    rows_idx = np.nonzero(column == 1)[0].tolist()
    print('rows_idx:', rows_idx)

    n_rows = len(rows_idx)
    print('n_rows:', n_rows)

    tmp_Y = deepcopy(dow.Y)
    tmp_Y[rows_idx, idx] = 0
    
    row = np.zeros((n_cols, ))
    row[0] = 1

    rows_perm = multiset_permutations(row)
    print('rows_perm:', rows_perm)

    discarded_dows = list()
    tmp_neighbours = list()
    cartesian = product(rows_perm, repeat=n_rows)
    for prod in cartesian:
        prod = np.array(prod).reshape((n_rows, n_cols))
        print('prod:', prod)
        prod_Y = deepcopy(tmp_Y)
        prod_Y[np.ix_(rows_idx, colms_idx)] = prod
        print('prod_Y:', prod_Y)

        neighbour_dow = DOW(dow.m_storages, dow.n_fields, dow.k_vehicles)
        neighbour_dow.X = tmp_X
        neighbour_dow.Y = prod_Y
        neighbour_dow.Z = tmp_Z

        modify_rhs_constrs(neighbour_dow, constrs)
        larp, is_fit = fit(larp, neighbour_dow)
        if is_fit:
            neighbour_dow.to_vector()
            tmp_neighbours.append([neighbour_dow, neighbour_dow.obj_value])
            print(neighbour_dow)
        else:
            neighbour_dow.to_vector()
            discarded_dows.append(neighbour_dow)
        
    if tmp_neighbours:
        dows, obj_vals = zip(*tmp_neighbours)
        idx_min_obj_val = obj_vals.index(min(obj_vals))
        return larp, (idx, dows[idx_min_obj_val], discarded_dows)
    return larp, (idx, None, discarded_dows)

        
def _change_status_to_open(larp, constrs, dow, tmp_X, idx) -> tuple:

    # ----------------------------------

    tmp_Z = deepcopy(dow.Z)

    F_last_pos = np.nonzero(tmp_Z[tmp_Z.shape[0]-1, :] == 1)[0][-1]
    last_route = list()

    last_route.append(tmp_Z.shape[0]-1)
    while F_last_pos != tmp_Z.shape[1]-1:
        F_last_pos = np.nonzero(tmp_Z[F_last_pos, :] == 1)[0][0]
        last_route.append(F_last_pos)

    last_position = last_route[-2]   

    tmp_Z[last_position, idx] = 1
    tmp_Z[last_position, tmp_Z.shape[1]-1] = 0
    tmp_Z[idx, tmp_Z.shape[1]-1] = 1 

    # ----------------------------------

    tmp_Y = deepcopy(dow.Y)
    n_rows, n_cols = tmp_Y.shape
    print('n_rows:', n_rows)

    discarded_dows = list()
    tmp_neighbours = list()

    X_idx = [i for i in np.nonzero(tmp_X)[0]]
    print('X_idx:', X_idx)

    n_cols = len(X_idx)
    print('n_cols:', n_cols)
    
    row = np.zeros((n_cols,))
    row[0] = 1

    rows_perm = multiset_permutations(row)
    print('rows_perm:', rows_perm)

    cartesian = product(rows_perm, repeat=n_rows)
    for prod in cartesian:
        prod = np.array(prod).reshape((n_rows, n_cols))
        tmp_Y[:, X_idx] = prod

        neighbour_dow = DOW(dow.m_storages, dow.n_fields, dow.k_vehicles)
        neighbour_dow.X = tmp_X
        neighbour_dow.Y = tmp_Y
        neighbour_dow.Z = tmp_Z

        modify_rhs_constrs(neighbour_dow, constrs)
        larp, is_fit = fit(larp, neighbour_dow)
        if is_fit:
            neighbour_dow.to_vector()
            tmp_neighbours.append([neighbour_dow, neighbour_dow.obj_value])
            print(neighbour_dow)
        else:
            neighbour_dow.to_vector()
            discarded_dows.append(neighbour_dow)

    if tmp_neighbours:
        dows, obj_vals = zip(*tmp_neighbours)
        idx_min_obj_val = obj_vals.index(min(obj_vals))
        return larp, (idx, dows[idx_min_obj_val], discarded_dows)
    return larp, (idx, None, discarded_dows)
