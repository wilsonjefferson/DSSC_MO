import numpy as np
from itertools import product

from gurobipy import GRB
from src.utils.utils_waterflow.dow import DOW
from src.larp import LARP


def create_larp(inputs:dict):
    larp = LARP(**inputs)
    
    # NOTE: Since it is difficult to find an optimal solution 
    # trying with random X, Y and Z (decision variables)
    # we fix a random X and set guroby to find and stop the very first feasible solution
    larp.model.setParam('OutputFlag', 0)
    larp.model.setParam('SolutionLimit', 1)
    larp.build()
    return larp

def remove_constrs(larp:LARP, constrs:dict) -> LARP:
    '''
        Remove addded constraints in LARP model, if they exist.

        Arguments
        ---------
        larp:LARP
        An instance of the LARP model

        constrs:dict
        Dictionary of constraint name and constraint formulation

        Return
        ------
        larp:LARP
        Same LARP instance without specified constraints
    '''

    if constrs is None:
        return larp

    model = larp.model

    constr = constrs.get('X_Constr_WaterFlow', None)
    if constr:
        X_len = larp.m_storages
        for i in range(X_len):
            model.remove(constr[i])

    constr = constrs.get('Y_Constr_WaterFlow', None)
    if constr:
        Y_rows, Y_cols = larp.n_fields, X_len
        for i, j in product(range(Y_rows), range(Y_cols)):
            model.remove(constr[i,j])

    constr = constrs.get('Z_Constr_WaterFlow', None)
    if constr:
        Z_rows, Z_cols = len(larp.J_0), len(larp.J_0)
        for u, v in product(range(Z_rows), range(Z_cols)):
            if u != v: # as the Z decision variable is defined in LARP
                model.remove(constr[u,v])
    
    model.update()
    return larp

def modify_rhs_constrs(dow:DOW, constrs:dict) -> None:
    '''
        Modify right-head-sides (RHS) for a set of constrains. 
        New RHS is given by the passed dow in input.

        Arguments
        ---------
        dow:DOW
        A drop-of-water (dow) representing a certain solution

        constrs:dict
        Dictionary of constraint name and constraint formulation

        Return
        ------
        None
    '''

    X_len = len(dow.X)
    constr = constrs['X_Constr_WaterFlow']
    for i in range(X_len):
        constr[i].rhs = dow.X[i]

    constr = constrs.get('Y_Constr_WaterFlow', None)
    if constr:
        Y_rows, Y_cols = dow.Y.shape
        for i, j in product(range(Y_rows), range(Y_cols)):
            constr[i,j].rhs = dow.Y[i,j]

    constr = constrs.get('Z_Constr_WaterFlow', None)
    if constr:
        Z_rows, Z_cols = dow.Z.shape
        for u, v in product(range(Z_rows), range(Z_cols)):
            if u != v: # as the Z decision variable is defined in LARP
                constr[u,v].rhs = dow.Z[u,v]
    
def add_constrs(larp:LARP, dow:DOW, option:str='all') -> tuple:
    '''
        Add new constraint(s) to the LARP model.

        Arguments
        ---------
        larp:LARP
        An instance of the LARP model

        dow:DOW
        A drop-of-water (dow) representing a certain solution

        option:str
        If equals to 'all', new constraints for X, Y and Z (LARP decision variables)
        are created; otherwise only new constraints for X are created

        Return
        ------
        larp:LARP
        Same LARP instance without specified constraints

        constrs:dict
        Dictionary of the created constraints
    '''

    model = larp.model
    constrs = dict()
    
    # if option is not equals to 'all', only X_Constr_WaterFlow constraints
    # are created and introduced in larp model
    X_len = len(dow.X)
    X_Constr_WaterFlow = model.addConstrs((larp.X[i] == dow.X[i] 
                    for i in range(X_len)), name='X_Constr_WaterFlow')
    constrs['X_Constr_WaterFlow'] = X_Constr_WaterFlow

    if option == 'all':
        Y_rows, Y_cols = dow.Y.shape
        Y_Constr_WaterFlow = model.addConstrs((larp.Y[i,j] == dow.Y[i,j] 
                        for i in range(Y_rows) 
                        for j in range(Y_cols)), name='Y_Constr_WaterFlow')
        constrs['Y_Constr_WaterFlow'] = Y_Constr_WaterFlow
        
        Z_rows, Z_cols = dow.Z.shape
        Z_Constr_WaterFlow = model.addConstrs((larp.Z[u,v] == dow.Z[u,v]
                        for u in range(Z_rows) 
                        for v in range(Z_cols)
                        if u!=v), name='Z_Constr_WaterFlow')
        constrs['Z_Constr_WaterFlow'] = Z_Constr_WaterFlow

    return larp, constrs

def fit(larp:LARP, dow:DOW) -> tuple:
    '''
        Execute the optimization of LARP model.

        Arguments
        ---------
        larp:LARP
        An instance of the LARP model

        dow:DOW
        A drop-of-water (dow) representing a certain solution

        Return
        ------
        larp: LARP
        The same LARP model instance

        bool
        True if a feasible solution is found, False otherwise
    '''

    model = larp.model
    model.setParam('OutputFlag', 0) # silent optimization logs

    # print('model optimization in-progress...')
    model.optimize()
    # print('model optimization COMPLETED')

    # print('model status:', model.status)
    # if ANY solution is found update dow
    if model.status in [GRB.SOLUTION_LIMIT, GRB.OPTIMAL]:
        dow.obj_value = model.ObjVal
        return larp, True
    return larp, False

def check_additional_constr(dow:DOW) -> bool:
    '''
        Additional check to control if drop-of-water (dow) 
        solution is feasible.

        Arguments
        ---------
        dow:DOW
        A drop-of-water (dow) representing a certain solution

        Return
        ------
        columns_nozero:bool
        True if additional check is satisfied, False otherwise
    '''

    X_idx = [i for i in np.nonzero(dow.X)[0]]
    columns_nozero = all([dow.Y[:, j].any() for j in X_idx])
    return columns_nozero
