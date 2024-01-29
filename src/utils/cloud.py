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

        no_feasible_dows = None
        generator = self.dows_generator(self.larp, self.max_pop)
        
        for dow, no_dows in generator:
            no_feasible_dows = deepcopy(no_dows)
            # print('make_rain | dow in generator:\n', dow)

            # print('check if dow in E_list...')
            no_E_list = dow not in E_list
            # print('check completed')

            # print('check if dow in discarded_list...')
            no_discarded_list = dow not in discarded_list
            # print('check completed')

            # print('check if dow in rainfall...')
            no_rainfall = dow not in rainfall
            # print('check completed.')

            if no_E_list and no_discarded_list and no_rainfall:
                rainfall.append(dow)

        discarded_dows.extend(no_feasible_dows)
        
        self.larp.model.resetParams()
        return rainfall, discarded_dows
    
    def dows_generator(self, larp:LARP, n_dows:int) -> tuple:
        discarded_dows = list()
        constrs = None
        for i in range(n_dows):
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

            dow.to_vector()
            yield dow, discarded_dows
        
        # print('remove additional constraints')
        remove_constrs(larp, constrs)
