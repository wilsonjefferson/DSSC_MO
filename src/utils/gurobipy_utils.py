import numpy as np
from itertools import product

from gurobipy import GRB
from src.utils.dow import DOW
from src.larp import LARP


def remove_constrs(larp:LARP, constrs:dict) -> LARP:

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
            if u != v:
                model.remove(constr[u,v])
    
    model.update()
    return larp

def modify_rhs_constrs(dow:DOW, constrs:dict) -> None:

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
            if u != v:
                constr[u,v].rhs = dow.Z[u,v]
    
def add_constrs(larp:LARP, dow:DOW, option:str='all') -> tuple:

    model = larp.model
    constrs = dict()
    
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

    model = larp.model
    model.setParam('OutputFlag', 0)

    model.update()
    # print('model optimization in-progress...')
    model.optimize()
    # print('model optimization COMPLETED')

    # print('model status:', model.status)
    if model.status in [GRB.SOLUTION_LIMIT, GRB.OPTIMAL]:
        dow.obj_value = model.ObjVal
        return larp, True
    return larp, False

def check_additional_constr(dow:DOW) -> bool:
    X_idx = [i for i in np.nonzero(dow.X)[0]]
    columns_nozero = all([dow.Y[:, j].any() for j in X_idx])
    return columns_nozero