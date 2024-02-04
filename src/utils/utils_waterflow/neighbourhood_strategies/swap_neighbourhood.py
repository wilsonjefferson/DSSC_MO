import numpy as np
from itertools import product
from copy import deepcopy

from src.utils.utils_waterflow.dow import DOW
from src.larp import LARP
from src.utils.gurobipy_utils import remove_constrs
from src.utils.utils_waterflow.neighbourhood_strategies.support_functions import feasibility_check, optimality_check


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

    dow.to_vector()
    print('dow:', dow)

    neighbours = list()
    discarded_dows = list()
    constrs = dict()

    X_idx_zeros = np.nonzero(dow.X == 0)[0]
    X_idx_nonzeros = np.nonzero(dow.X)[0]

    if len(X_idx_zeros) == 0 or len(X_idx_nonzeros) == 0:
        return dow, list(), discarded_dows
    
    cartesian = product(X_idx_zeros, X_idx_nonzeros)
    for zero_idx, nonzero_idx in cartesian:

        zero_idx += 1
        nonzero_idx += 1
        print('zero_idx:', zero_idx, 'nonzero_idx:', nonzero_idx)

        # adjust X decision variable
        tmp_X = deepcopy(dow.X)
        tmp_X[zero_idx-1] = 1
        tmp_X[nonzero_idx-1] = 0

        # adjust Y decision variable
        tmp_Y = deepcopy(dow.Y)
        reassign_indexes = np.nonzero(tmp_Y == nonzero_idx)[0]
        tmp_Y[reassign_indexes] = zero_idx
        print('tmp_Y:', tmp_Y)

        # adjust Z decision variable
        tmp_Z = deepcopy(dow.Z)
        reassign_indexes = np.nonzero(tmp_Z == nonzero_idx)[0]
        tmp_Z[reassign_indexes] = zero_idx
        print('tmp_Z:', tmp_Z)

        feasibility_check(dow.m_storages, dow.n_fields, dow.k_vehicles, 
                          tmp_X, tmp_Y, tmp_Z, larp, constrs, 
                          neighbours, discarded_dows)

    # print('neighbours:', neighbours)
    larp = remove_constrs(larp, constrs)
    
    local_optimum, dows = optimality_check(dow, neighbours)
    return local_optimum, dows, discarded_dows
