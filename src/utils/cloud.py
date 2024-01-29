from copy import deepcopy

from src.larp import LARP
from src.utils.dow import DOW

from src.utils.gurobipy_utils import (add_constrs, 
                                      modify_rhs_constrs, 
                                      remove_constrs, fit,
                                      check_additional_constr)

class CLOUD:

    def __init__(self, larp, max_pop) -> None:
        self.larp = larp
        self.max_pop = max_pop # n_dows

    def make_rain(self, E_list:list, discarded_list:list) -> tuple:
        discarded_dows = list()
        rainfall = list()
        self.larp.model.setParam('SolutionLimit', 1)

        n_dows = self.max_pop
        while True:
            no_feasible_dows = None
            generator = self.dows_generator(self.larp, n_dows)
            
            for dow, no_dows in generator:
                no_feasible_dows = deepcopy(no_dows)
                if dow not in E_list and dow not in discarded_list and dow not in rainfall:
                    rainfall.append(dow)

            n_dows = self.max_pop - len(rainfall)
            # print('n_dows:', n_dows)
            if n_dows == 0:
                discarded_dows.extend(no_feasible_dows)
                break
        
        self.larp.model.resetParams()
        return rainfall, discarded_dows
    
    def dows_generator(self, larp:LARP, n_dows:int) -> tuple:
        discarded_dows = list()
        constrs = None
        for _ in range(n_dows):
            while True:
                dow = DOW(larp.m_storages, larp.n_fields, larp._k_vehicles)
                dow.set_rand_X()
                # print('dow created')

                if constrs:
                    modify_rhs_constrs(dow, constrs)
                else:
                    larp, constrs = add_constrs(larp, dow, None)

                larp, is_fit = fit(larp, dow)
                
                pass_additional_constr = False
                if is_fit:
                    dow.Y = larp.Y
                    dow.Z = larp.Z
                    pass_additional_constr = check_additional_constr(dow)

                if is_fit and pass_additional_constr:
                    # print('iter:', i, ' --> DOW FEASIBLE', sep=' ')
                    larp.model.reset(0)
                    break
                
                discarded_dows.append(dow)
                # print('iter:', i, ' --> DOW NOT FEASIBLE', sep=' ')
                larp.model.reset(0)

            yield dow, discarded_dows
        
        # print('remove additional constraints')
        remove_constrs(larp, constrs)
