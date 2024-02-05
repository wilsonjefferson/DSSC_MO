import numpy as np
import functools as ft
import multiprocessing as mp
from itertools import product
from copy import deepcopy
from gurobipy import GRB

from src.utils.utils_waterflow.dow import DOW
from src.larp import LARP
from src.utils.gurobipy_utils import add_constrs, remove_constrs
from src.utils.gurobipy_utils import modify_rhs_constrs

from src.utils.utils_waterflow.neighbourhood_strategies.support_functions import optimality_check


N_PROCESSES = mp.cpu_count()


class OPT1:

    def __init__(self, larp:LARP, dow:DOW):
        self._larp = larp
        self._dow = dow

    @property
    def dow(self):
        return self._dow

    @dow.setter
    def dow(self, other:DOW):
        self._dow = other

    @property
    def larp(self):
        return self._larp

    @larp.setter
    def larp(self, other):
        self._larp = other

    def search(self):
        # print('opt_1 | self._dow vector:', self._dow)
        self._dow.to_matrix()
        self.larp, self.constrs = add_constrs(self._larp, self._dow)
        self._dow.to_vector()
        
        neighbours = list()
        discarded_list = list()
        optimal_neighbours = list()

        for idx in range(len(self._dow.X)):
            tmp_X = deepcopy(self._dow.X)
            tmp_X[idx] = not self._dow.X[idx]
            dow_new_status_to_close = tmp_X[idx] == 0
            idx += 1 # adjust index of X to match the values in Y and Z

            if dow_new_status_to_close: # binary value is 0
                # change binary status to 1 (open)
                tmp = self._change_status_to_close(tmp_X, idx)
            else: # binary value is 1
                # change binary status to 0 (close)
                tmp = self._change_status_to_open(tmp_X, idx)
            
            good_neighbour, other_neighbours, discarded_dows = tmp

            optimal_neighbours.append(good_neighbour)
            neighbours.extend(other_neighbours)
            discarded_list.extend(discarded_dows)

        self.larp = remove_constrs(self.larp, self.constrs) # remove added constraints, no more needed

        if optimal_neighbours: # i.e. current optimal has a list of neighbour solutions
            # retrieve the best neighbour according the objective value
            obj_vals = [neighbour.obj_value for neighbour in optimal_neighbours]
            idx_min = obj_vals.index(min(obj_vals))
            candidate = optimal_neighbours[idx_min]
            dows = neighbours
        else:
            candidate = self._dow
            dows = []

        # compare current self._dow vs candidate self._dow for the role of local optimum
        if self._dow.obj_value <= candidate.obj_value:
            # print('local optimum is self._dow!')
            local_optimum = self._dow
            dows.extend(optimal_neighbours)
        else:
            # print('local optimum is candidate!')
            local_optimum = candidate
            optimal_neighbours.remove(candidate)
            neighbours.extend(optimal_neighbours)
            dows = neighbours
        
        return local_optimum, dows, discarded_list

    def _change_status_to_close(self, tmp_X:np.ndarray, idx:int) -> tuple:
        '''
            If change status from 1 (open) to 0 (close), following routine
            is executed to adjust Y and Z attributes and generate new dows.
            These dows are evaluated against the LARP model to determine which
            dows are feasible. 

            Arguments
            ---------
            larp:LARP
            An instance of the LARP model

            constrs:dict
            Dictionary of additional constrains for LARP model

            dow:DOW
            A drop-of-water (dow) representing a certain solution

            tmp_X:np.ndarray
            A copy of dow.X array

            idx:int
            Integer number representing the position of the changed binary value

            Return
            ------
            local_optimum:DOW
            New local optimum solution

            dows:list
            List of neighbours dows of the local optimum

            discarded_dows:list
            List of no feasible solutions
        '''

        discarded_dows = list()
        tmp_neighbours = list()
        
        # a no-really elegant way to adjust a copy of dow.Z
        # to match the changes from dow.X 
        self.tmp_dow = deepcopy(self._dow)

        # adapt decision variable Z
        unwanted_value_idx = np.nonzero(self.tmp_dow.Z == idx)[0]
        if unwanted_value_idx+1 < len(self.tmp_dow.Z):
            if self.tmp_dow.Z[unwanted_value_idx-1] == self.tmp_dow.Z[unwanted_value_idx+1]:
                self.tmp_dow.Z = np.delete(self.tmp_dow.Z, unwanted_value_idx+1)
        np.delete(self.tmp_dow.Z, unwanted_value_idx)
        if self.tmp_dow.Z[-1] == 0:
            self.tmp_dow.Z = np.delete(self.tmp_dow.Z, -1)

        # adapt decision variable Y
        self.indexes_positions_of_idx = np.nonzero(self._dow.Y == idx)
        uniques = np.unique(self._dow.Y)
        acceptable_values = np.nonzero(uniques != idx)
        acceptable_values = uniques[acceptable_values]

        cartesian = product(acceptable_values, repeat=len(self.indexes_positions_of_idx))

        self.feasibility_params = [tmp_X, self.tmp_dow.Z, tmp_neighbours, discarded_dows]

        print('start parallelization...')
        with mp.Pool(N_PROCESSES) as pool:
            pool.map(self._task_to_close, cartesian)
        print('parallelization completed.')

        local_optimum, dows = optimality_check(self._dow, tmp_neighbours)
        return local_optimum, dows, discarded_dows 
    
    def _task_to_close(self, disp:tuple):
        self.tmp_dow.Y[self.indexes_positions_of_idx] = disp
        tmp_Y = deepcopy(self.tmp_dow.Y)
        self.feasibility_params.insert(1, tmp_Y)
        self.feasibility_check()

    def _change_status_to_open(self, tmp_X:np.ndarray, idx:int) -> tuple:
        '''
            If change status from 0 (close) to 0 (open), following routine
            is executed to adjust Y and Z attributes and generate new dows.
            These dows are evaluated against the LARP model to determine which
            dows are feasible. 

            Arguments
            ---------
            larp:LARP
            An instance of the LARP model

            constrs:dict
            Dictionary of additional constrains for LARP model

            dow:DOW
            A drop-of-water (dow) representing a certain solution

            tmp_X:np.ndarray
            A copy of dow.X array

            idx:int
            Integer number representing the position of the changed binary value

            Return
            ------
            local_optimum:DOW
            New local optimum solution

            dows:list
            List of neighbours dows of the local optimum

            discarded_dows:list
            List of no feasible solutions
        '''

        discarded_dows = list()
        tmp_neighbours = list()

        # adapt decision variable Z
        tmp_Z = deepcopy(self._dow.Z)

        if len(tmp_Z) == 0 or tmp_Z is None:
            tmp_Z = np.array([0, idx])
        else:
            tmp_Z = np.append(tmp_Z, idx)

        # adapt decision variable Y
        uniques = np.unique(self._dow.Y)
        uniques = np.append(uniques, idx)
        cartesian = product(uniques, repeat=len(self._dow.Y))
            
        self.feasibility_params = [tmp_X, tmp_Z, tmp_neighbours, discarded_dows]

        print('start parallelization...')
        with mp.Pool(N_PROCESSES) as pool:
            pool.map(self._task_to_open, cartesian)
        print('parallelization completed.')

        local_optimum, dows = optimality_check(self._dow, tmp_neighbours)
        return local_optimum, dows, discarded_dows

    def _task_to_open(self, disp:tuple):
        tmp_Y = np.array(disp)
        self.feasibility_params.insert(1, tmp_Y)
        self.feasibility_check()

    def feasibility_check(self) -> None:
        '''
        This is a suppot function used to create a new dow solution 
        and check if this solution is feasible.

        Arguments
        ---------
        m_storages:int
        Integer number of storages

        n_fields:int
        Integer number of fields

        k_vehicles:int
        Integer number of vehicles

        tmp_X:np.array
        Temporary X decision variable to create a new dow

        tmp_Y:np.array
        Temporary Y decision variable to create a new dow
        
        tmp_Z:np.array
        Temporary Z decision variable to create a new dow
        
        larp:LARP
        An instance of the LARP model
        
        constrs:dict
        Dictionary of additional constrains for LARP model
        
        tmp_neighbours:list
        List of temporary neighbours 
        
        discarded_dows:list
        List if discarded dows since they are not feasible
        
        Return
        ------
        None
        '''

        tmp_X, tmp_Y, tmp_Z, tmp_neighbours, discarded_dows = self.feasibility_params
        neighbour_dow = DOW(self._dow.m_storages, self._dow.n_fields, self._dow.k_vehicles)
        neighbour_dow.X = tmp_X
        neighbour_dow.Y = tmp_Y
        neighbour_dow.Z = tmp_Z

        neighbour_dow.to_matrix()
        # print('neighbour dow:', neighbour_dow)

        modify_rhs_constrs(neighbour_dow, self.constrs)

        # print('fitting LARP model...')
        is_fit = self.fit(neighbour_dow)
        # print('fitting completed')

        neighbour_dow.to_vector()

        if is_fit:
            # print('neighbour dow is FEASIBLE')
            tmp_neighbours.append([neighbour_dow, neighbour_dow.obj_value])
        else:
            # print('neighbour dow is NOT FEASIBLE')
            discarded_dows.append(neighbour_dow)

    def fit(self, dow:DOW) -> bool:
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

        model = self._larp.model
        model.setParam('OutputFlag', 0) # silent optimization logs

        # print('model optimization in-progress...')
        model.optimize()
        # print('model optimization COMPLETED')

        # print('model status:', model.status)
        # if ANY solution is found update dow
        if model.status in [GRB.SOLUTION_LIMIT, GRB.OPTIMAL]:
            dow.obj_value = model.ObjVal
            return True
        return False
    