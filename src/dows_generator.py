from gurobipy import GRB
from src.dow import DOW
from src.larp import LARP
import numpy as np

def check_and_fit(larp:LARP, dow:DOW):

    model = larp.model
    model.addConstrs((larp.X[i] == dow.X[i] for i in range(len(dow.X))), name='Constr_WaterFlow')

    model.optimize()
    print('LARP model optimized')

    for i in range(len(dow.X)):
        constr = model.getConstrByName(f'Constr_WaterFlow[{i}]')
        model.remove(constr)

    print('LARP status :', model.status)
    if model.status in [GRB.SOLUTION_LIMIT, GRB.OPTIMAL]:
        dow.obj_value = model.ObjVal
        dow.Y = larp.Y
        dow.Z = larp.Z 
        return check_additional_constr(dow)
    return False

def check_additional_constr(dow:DOW) -> bool:
    X_idx = [i for i in np.nonzero(dow.X)[0]]
    columns_nozero = all([dow.Y[:, j].any() for j in X_idx])
    return columns_nozero

def dows_generator(larp, n_dows:int, m_storages:int, n_fields:int, k_vehicles:int):
    for i in range(n_dows):
        print('generator loop:', i)
        while True:
            dow = DOW(m_storages, n_fields, k_vehicles)
            dow.set_rand_X()
            print('dow created')
            
            if check_and_fit(larp, dow):
                print('iter:', i, ' --> DOW FEASIBLE', sep=' ')
                break
            else:
                print('iter:', i, ' --> DOW NOT FEASIBLE', sep=' ')
            
            larp.model.reset(1)
        yield dow
