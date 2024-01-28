from itertools import product

from gurobipy import GRB
from src.dow import DOW
from src.larp import LARP


def remove_constrs(larp:LARP, constrs:dict) -> None:

    model = larp.model

    X_len = larp.m_storages
    Y_rows, Y_cols = larp.n_fields, X_len
    Z_rows, Z_cols = len(larp.J_0), len(larp.J_0)

    constr = constrs['X_Constr_WaterFlow']
    for i in range(X_len):
        model.remove(constr[i])

    constr = constrs['Y_Constr_WaterFlow']
    for i, j in product(range(Y_rows), range(Y_cols)):
        model.remove(constr[i,j])

    constr = constrs['Z_Constr_WaterFlow']
    for u, v in product(range(Z_rows), range(Z_cols)):
        if u != v:
            model.remove(constr[u,v])
    
    model.update()

def modify_rhs_constrs(dow:DOW, constrs:dict) -> None:

    X_len = len(dow.X)
    Y_rows, Y_cols = dow.Y.shape
    Z_rows, Z_cols = dow.Z.shape

    constr = constrs['X_Constr_WaterFlow']
    for i in range(X_len):
        constr[i].rhs = dow.X[i]

    constr = constrs['Y_Constr_WaterFlow']
    for i, j in product(range(Y_rows), range(Y_cols)):
        constr[i,j].rhs = dow.Y[i,j]

    constr = constrs['Z_Constr_WaterFlow']
    for u, v in product(range(Z_rows), range(Z_cols)):
        if u != v:
            constr[u,v].rhs = dow.Z[u,v]
    
def add_constrs(larp:LARP, dow:DOW) -> dict:

    model = larp.model

    X_len = len(dow.X)
    Y_rows, Y_cols = dow.Y.shape
    Z_rows, Z_cols = dow.Z.shape

    constrs = dict()

    X_Constr_WaterFlow = model.addConstrs((larp.X[i] == dow.X[i] 
                    for i in range(X_len)), name='X_Constr_WaterFlow')
    constrs['X_Constr_WaterFlow'] = X_Constr_WaterFlow
    
    Y_Constr_WaterFlow = model.addConstrs((larp.Y[i,j] == dow.Y[i,j] 
                    for i in range(Y_rows) 
                    for j in range(Y_cols)), name='Y_Constr_WaterFlow')
    constrs['Y_Constr_WaterFlow'] = Y_Constr_WaterFlow
    
    Z_Constr_WaterFlow = model.addConstrs((larp.Z[u,v] == dow.Z[u,v]
                    for u in range(Z_rows) 
                    for v in range(Z_cols)
                    if u!=v), name='Z_Constr_WaterFlow')
    constrs['Z_Constr_WaterFlow'] = Z_Constr_WaterFlow

    return constrs

def fit(larp:LARP, dow:DOW) -> bool:

    model = larp.model

    print('model optimization in-progress...')
    model.optimize()
    print('model optimization COMPLETED')

    # print('model status:', model.status)
    if model.status == GRB.OPTIMAL:
        dow.obj_value = model.ObjVal
        return True
    return False