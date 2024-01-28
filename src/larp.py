import pandas as pd
import numpy as np

import gurobipy as gp
from gurobipy import GRB

gp.disposeDefaultEnv() # reset default env variables


class LARP:

    def __init__(self, 
                 facility:str, 
                 k_vehicles:int, 
                 Q_vehicle_capacity:int, 
                 fields:list, 
                 storages:list, 
                 households:list, 
                 f:dict, 
                 q:dict, 
                 fs_dist:pd.DataFrame, 
                 cs_dist:pd.DataFrame, 
                 pivot_d:pd.DataFrame) -> None:
        '''
            This class rapresent the LARP model, it is composed
            by a set of class methods to define the decision variables,
            the objective function and the constrains.

            Arguments
            ---------

            facility:str
            Name of the main facility

            k_vehicles:int
            Number of vehicles which transport the agricultural wastes
            from the storages to the main facility

            Q_vehicle_capacity:int
            Capacity of the k_vehicles in tons

            fields:list
            List of field labels

            storages:list
            List of storage labels

            households:list
            List of households labels

            f:dict
            Dictionary of storages and cost

            q:dict
            Dictionary of storages and storage capacities

            fs_dist:pd.DataFrame
            storage to storage distance dataframe, also the main facility
            is included

            cs_dist:pd.DataFrame
            field to storage distance dataframe, it is a symmetric
            dataframe

            pivot_d:pd.DataFrame
            Dataframe rapresenting the amount of agricultural waste per fieds
            
        '''

        self._X = None
        self._Y = None
        self._Z = None       

        self._X_sol = None
        self._Y_sol = None
        self._Z_sol = None

        self._facility = facility
        self._households = households

        self._k_vehicles = k_vehicles
        self._Q_vehicle_capacity = Q_vehicle_capacity

        self._fields = fields
        self.n_fields = len(fields)

        self.m_storages = len(storages)
        self._storages = storages
        
        self.J_0 = storages + [facility]

        self._f = f
        self._q = q

        self._fs_dist = fs_dist
        self._cs_dist = cs_dist
        self._pivot_d = pivot_d

        self.fields_idx = dict(zip(fields, range(self.n_fields)))
        self.storages_idx = dict(zip(storages, range(self.m_storages)))
        self.idx_to_storages = dict(zip(range(self.m_storages), storages))
        self.J_0_idx = dict(zip(self.J_0, range(len(self.J_0))))

        # larp model
        self._model = gp.Model('location_assignment_routing_problem') # general Gurobi mdodel
        self._model.modelSense = GRB.MINIMIZE # decleare the problem as minimization problem
        self._model.setParam('outputFlag', 0)

    def __del__(self):
        self.dispose()

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y
    
    @property
    def Z(self):
        return self._Z
    
    @property
    def X_sol(self):
        return self._X_sol
    
    @property
    def Y_sol(self):
        return self._Y_sol
    
    @property
    def Z_sol(self):
        return self._Z_sol
    
    @property
    def model(self):
        return self._model
    
    @property
    def inputs(self):
        return {'facility': self._facility,
            'households': self._households,
            'k_vehiccles': self._k_vehicles,
            'Q_vehicle_capacity': self._Q_vehicle_capacity,
            'fields': self._fields,
            'storages': self._storages,
            'f': self._f,
            'q': self._q,
            'fs_dist': self._fs_dist,
            'cs_dist': self._cs_dist,
            'pivot_d': self._pivot_d}

    def _declare_decision_variables(self) -> None:
        '''
            Support function used to decleare the decision variables
        '''
        self._X = self._model.addVars([j for j in range(len(self.J_0))], vtype=GRB.BINARY, name='X')
        self._Y = self._model.addVars([(i,j) for i in range(self.n_fields) for j in range(self.m_storages)], 
                                          vtype=GRB.BINARY, name='Y')
        self._Z = self._model.addVars([(u,v) for u in range(len(self.J_0)) for v in range(len(self.J_0)) if u!=v], 
                                          vtype=GRB.BINARY, name='Z')

    def _decleare_objective_function(self) -> None:
        '''
            Support function to decleare the objective function of the model
        '''
        self._model.setObjective( 
            gp.quicksum(self._f[j]*self._X[self.storages_idx[j]] for j in self._storages) +
            gp.quicksum(self._cs_dist.loc[i,j]*self._pivot_d.loc[i,h]*self._Y[self.fields_idx[i],self.storages_idx[j]] 
                        for i in self._fields for j in self._storages for h in self._households) +
            gp.quicksum(self._fs_dist.loc[u,v]*self._Z[self.J_0_idx[u],self.J_0_idx[v]] 
                        for u in self.J_0 for v in self.J_0 if u!=v)
        )

    def _decleare_constrains(self) -> None:
        '''
            Support function to decleare the set of model constraints.
        '''

        # all fields have to be assigned to exactly one storage
        for i in self._fields:
            self._model.addConstr(
                gp.quicksum(self._Y[self.fields_idx[i],self.storages_idx[j]] for j in self._storages) == 1)
        
        # (capacity constraint) amount of agricultural waste cannot exceed the
        # storage capacity, this is valid for each storage
        for j in self._storages:
            self._model.addConstr(
                gp.quicksum(self._pivot_d.loc[i,h]*self._Y[self.fields_idx[i],self.storages_idx[j]] 
                            for i in self._fields for h in self._households) <= self._q[j]*self._X[self.storages_idx[j]])
        
        # (conservatibe constrains) k_vehicles leave the main facility
        # and k_vehicles return to the main facility
        self._model.addConstr(
            gp.quicksum(self._Z[self.storages_idx[u],self.J_0_idx[self._facility]] 
                        for u in self._storages) == self._k_vehicles)
        self._model.addConstr(
            gp.quicksum(self._Z[self.J_0_idx[self._facility],self.storages_idx[v]] 
                        for v in self._storages) == self._k_vehicles)

        # linearization of non-linear constrains
        self._apply_linearization()

        # include constrains to eliminate subtorus
        self._eliminate_subtours()

        # NOTE: Totally Unimodularity constraint
        # Since the adjacency matrix of the distances among the storages
        # is total unimodular, it means the LARP decision variables are all integer
        # therefore, the following set of constrains can be included
        for i in self._fields:
            for j in self._storages:
                self._model.addConstr(self._Y[self.fields_idx[i],self.storages_idx[j]] >= 0)
    
    def _eliminate_subtours(self) -> None:
        '''
            Support functino to include additional constraints
            which ensure no subtour is acceptable
        '''
        # auxiliary decision variables
        T = self._model.addVars([w for w in range(self.m_storages)], vtype=GRB.INTEGER, name='T')

        for u in self._storages:
            self._model.addConstr(T[self.storages_idx[u]] >= 0)

        for u in self._storages:
            for v in self._storages:
                if u != v:
                    self._model.addConstr(T[self.storages_idx[u]] - T[self.storages_idx[v]] + 
                                               self._Q_vehicle_capacity*self._Z[self.storages_idx[u], self.storages_idx[v]] <= 
                            self._Q_vehicle_capacity - (self._k_vehicles**(-1))*gp.quicksum(
                                self._pivot_d.loc[i,h]*self._Y[self.fields_idx[i],self.storages_idx[v]] 
                                                                                for i in self._fields for h in self._households))

            self._model.addConstr(T[self.storages_idx[u]] <= self._Q_vehicle_capacity)
            self._model.addConstr((self._k_vehicles**(-1))*gp.quicksum(
                self._pivot_d.loc[i,h]*self._Y[self.fields_idx[i],self.storages_idx[u]] 
                                                                for i in self._fields for h in self._households) <= T[self.storages_idx[u]])
    
    def _apply_linearization(self) -> None:
        '''
            Support function to apply linearization on non-linear
            constraints.
        '''
        # auxiliary decision variables
        W_1 = self._model.addVars([(u,v) for u in range(len(self.J_0)) for v in range(len(self.J_0)) if u!=v], 
                                       vtype=GRB.BINARY, name='W_1')
        W_2 = self._model.addVars([(u,v) for u in range(len(self.J_0)) for v in range(len(self.J_0)) if u!=v], 
                                       vtype=GRB.BINARY, name='W_2')

        for v in self._storages:
            self._model.addConstr(gp.quicksum(W_1[self.J_0_idx[u], self.storages_idx[v]] 
                                                   for u in self.J_0 if u!=v) == self._X[self.storages_idx[v]])

        for u in self.J_0:
            for v in self.J_0:
                if u!=v:
                    self._model.addConstr(W_1[self.J_0_idx[u], self.J_0_idx[v]] <= self._X[self.J_0_idx[u]])
                    self._model.addConstr(
                        W_1[self.J_0_idx[u], self.J_0_idx[v]] >= self._X[self.J_0_idx[u]] + self.Z[self.J_0_idx[u], self.J_0_idx[v]] - 1
                    )
                    self._model.addConstr(W_1[self.J_0_idx[u], self.J_0_idx[v]] <= self._Z[self.J_0_idx[u], self.J_0_idx[v]])

        for u in self._storages:
            self._model.addConstr(gp.quicksum(W_2[self.storages_idx[u], self.J_0_idx[v]] 
                                                   for v in self.J_0 if u!=v) == self._X[self.storages_idx[u]])

        for u in self.J_0:
            for v in self.J_0:
                if u!=v:
                    self._model.addConstr(W_2[self.J_0_idx[u], self.J_0_idx[v]] <= self._X[self.J_0_idx[v]])
                    self._model.addConstr(
                        W_2[self.J_0_idx[u], self.J_0_idx[v]] >= self._X[self.J_0_idx[v]] + self._Z[self.J_0_idx[u], self.J_0_idx[v]] - 1
                    )
                    self._model.addConstr(W_2[self.J_0_idx[u], self.J_0_idx[v]] <= self._Z[self.J_0_idx[u], self.J_0_idx[v]])

    def build(self) -> None:
        '''
            Public method to build the model, this process is composed
            by the declaration of decision variables, declaration of 
            objective function and declaration of constraints. This is a
            required step to set the LARP model and start the optimization.
        '''
        self._declare_decision_variables()
        # print('LARP decision variables defined')

        self._decleare_objective_function()
        # print('LARP objective function defined')

        self._decleare_constrains()
        # print('LARP constrains defined')

        print('-- LARP model build COMPLETED --')

    def optimize(self) -> None:
        '''
            Public method to start the LARP optimization
        '''
        self._model.optimize()

    def get_solutions(self) -> tuple:
        '''
            Public function to return the model decision variables
            with a numpy matrix shape.

            Arguments
            ---------

            None

            Return
            ------

            tuple
            Tuple of X, Y, and Z solution of the LARP model with a
            matrix shape
        '''
        self._X_sol = np.array([self._X[self.storages_idx[j]].x for j in self._storages])
        X_sol_rapresentation = [self.idx_to_storages[x] for x in range(len(self.X_sol)) if self._X_sol[x] > 0.5]

        self._Y_sol = np.array([[self._Y[self.fields_idx[i],self.storages_idx[j]].x for j in self._storages] for i in self._fields])
        Y_sol_rapresentation = pd.DataFrame(self._Y_sol, columns=self._storages, index=self._fields)

        self._Z_sol = np.array([[self._Z[self.J_0_idx[u],self.J_0_idx[v]].x if u!=v else 0.0 for v in self.J_0] for u in self.J_0])
        Z_sol_rapresentation = pd.DataFrame(self._Z_sol, columns=self._fs_dist.columns, index=self._fs_dist.index)
        
        return X_sol_rapresentation, Y_sol_rapresentation, Z_sol_rapresentation

    def get_objvalues(self) -> dict:
        '''
            Public method to return meaningful information
            about the LARP solution.

            Arguments
            ---------

            None

            Return
            ------

            results:dict
            A dictionary with the following information:
              - larp_onjval: LARP objective value for the discovered solution
              - location_cost: costs to open certain storages
              - assignement_cost: costs to assign fields to certain storages
              - transportation_cost: costs to transport agricultural wastes
              - runtime: amount of time needed to find optimal solution

        '''
        try:
            larp_model_objval = round(self._model.ObjVal, 2)
        except AttributeError as e:
            print('WARNING: problem is infeasible')
            larp_model_objval, location_cost, assignment_cost, transportation_cost = None, None, None, None
        else:
            location_cost = sum(self._f[j]*self._X_sol[self.storages_idx[j]] for j in self._storages) 
            assignment_cost = sum(self._cs_dist.loc[i,j]*self._pivot_d.loc[i,h]*self._Y_sol[self.fields_idx[i],self.storages_idx[j]] 
                                for i in self._fields for j in self._storages for h in self._households) 
            transportation_cost = sum(self._fs_dist.loc[u,v]*self._Z_sol[self.J_0_idx[u],self.J_0_idx[v]] 
                                    for u in self.J_0 for v in self.J_0 if u!=v)

            location_cost = round(location_cost, 2)
            assignment_cost = round(assignment_cost, 2)
            transportation_cost = round(transportation_cost, 2)

            total_cost = location_cost+assignment_cost+transportation_cost
            total_cost = round(total_cost, 2)

            # just to check if the sum of partial cost is equal to the objval
            assert larp_model_objval == total_cost, \
                f'ERROR: ObjVal = {larp_model_objval}, total cost = {total_cost}'

        results = {'larp_objval':larp_model_objval, 'location_cost':location_cost, 
                   'assignment_cost':assignment_cost, 'transportation_cost':transportation_cost, 
                   'runtime':round(self._model.Runtime, 2)}
        return results
    
    def get_execution_time(self) -> float:
        '''
            Public method to get the execution time.

            Arguments
            ---------

            None

            Return
            ------

            float
            Runtime rounded
        '''
        return round(self._model.Runtime, 2)
    
    def dispose(self) -> None:
        '''
            Public method to dispose the LARP model

            Arguments
            ---------

            None

            Return
            ------

            None
        '''
        self._model.dispose()
        gp.disposeDefaultEnv()