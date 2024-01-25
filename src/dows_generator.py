from gurobipy import GRB
from src.dow import DOW
from src.larp import LARP


def check_and_fit(larp:LARP, dow:DOW):

    model = larp.model
    model.addConstrs((larp.X[i] == dow.X[i] for i in range(len(dow.X))), name='Constr_WaterFlow')

    model.optimize()
    print('LARP model optimized')

    for i in range(len(dow.X)):
        constr = model.getConstrByName(f'Constr_WaterFlow[{i}]')
        model.remove(constr)

    if model.status == GRB.SOLUTION_LIMIT:
        dow.obj_value = model.ObjVal
        return True, larp.Y, larp.Z
    return False, None, None

def dows_generator(larp, n_dows:int, m_storages:int, n_fields:int, k_vehicles:int):
    for i in range(n_dows):
        print('generator loop:', i)
        while True:
            dow = DOW(m_storages, n_fields, k_vehicles)
            print('dow created')
            is_feasible, Y, Z =  check_and_fit(larp, dow)

            if is_feasible:
                dow.Y = Y
                dow.Z = Z 
                print('iter:', i, ' --> DOW FEASIBLE', sep=' ')
                break
            else:
                print('iter:', i, ' --> DOW NOT FEASIBLE', sep=' ')
            
            larp.model.reset(1)
        yield dow
