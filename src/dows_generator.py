from gurobipy import GRB
from src.dow import DOW
from src.larp import LARP


def check_and_fit(larp:LARP, dow:DOW):

    larp._declare_decision_variables()
    larp._decleare_objective_function()
    larp._decleare_constrains()

    model = larp.model

    for i in range(len(dow.X)):
        model.addConstr(larp.X[i] == dow.X[i])

    rows, cols = dow.Y.shape
    for i in range(rows):
        for j in range(cols):
            model.addConstr(larp.Y[i,j] == dow.Y[i,j])
 
    rows, cols = dow.Z.shape
    for u in range(rows):
        for v in range(cols):
            if u == v:
                if dow.Z[u,v] == 1:
                    return False
            else:
                model.addConstr(larp.Z[u,v] == dow.Z[u,v])

    model.optimize()

    if model.status == GRB.OPTIMAL:
        dow.obj_value = model.ObjVal
        return True
    return False

def dows_generator(larp, n_dows:int, m_storages:int, n_fields:int, k_vehicles:int):
    for i in range(n_dows):
        while True:
            dow = DOW(m_storages, n_fields, k_vehicles)
            if check_and_fit(larp, dow):
                print('iter:', i, ' --> DOW FEASIBLE', sep=' ')
                break
            else:
                print('iter:', i, ' --> DOW NOT FEASIBLE', sep=' ')
        yield dow
