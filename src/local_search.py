from copy import deepcopy

from src.larp import LARP
from src.utils.dow import DOW
# from src.utils.opt_1_neighbourhood import opt_1
from src.utils.swap_neighbourhood import swap


def local_search(larp:LARP, dow:DOW) -> tuple:
    
    local_optimum = dow
    excluded_dows = list()
    discarded_dows = list()
    neighbours = None
    
    while True:
        solution, neighbours, no_feasible_dows = swap(larp, local_optimum)
        discarded_dows.extend(no_feasible_dows)

        if solution == local_optimum:
            break
        
        local_optimum.to_vector()
        excluded_dows.extend([local_optimum])
        excluded_dows.extend(neighbours)
        
        local_optimum = solution
    
    local_optimum.to_vector()
    return local_optimum, neighbours, excluded_dows, discarded_dows
