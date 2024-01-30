from collections import Counter

from src.larp import LARP
from src.utils.utils_waterflow.local_search import local_search, erosion

from src.utils.utils_waterflow.dow import DOW
from src.utils.utils_waterflow.clouds_generator import clouds_generator


def waterflow(larp:LARP, max_cloud:int, max_pop:int, max_UIE:int, min_ero:int) -> DOW:

    optimal_dows = dict()
    P0_list = list()
    UE_list = list()
    E_list = list()

    excluded_list = list()
    discarded_list = list()

    clouds = clouds_generator(max_cloud, larp, max_pop)

    for cloud in clouds:
        rainfall, discarded_dows = cloud.make_rain(E_list, discarded_list)
        discarded_list.extend(discarded_dows)

        for dow in rainfall:
            local_optimum, neighbours, excluded_dows, discarded_dows = local_search(larp, dow)
            excluded_list.extend(excluded_dows)
            discarded_list.extend(discarded_dows)
            UE_list.append(local_optimum) # for erosion process
        
            if local_optimum not in optimal_dows.keys():
                optimal_dows[local_optimum] = neighbours
        
        dow_occurances = Counter(UE_list)
        dow_occurances = [dow for dow, occurs in dow_occurances.items() if occurs >= min_ero]
        for dow in dow_occurances:
            
            neighbours = optimal_dows[dow]

            tmp = erosion(larp, dow, neighbours, max_UIE,
                excluded_list, discarded_list, optimal_dows, 
                P0_list, UE_list, E_list)
            
            dow_optimum, _, excluded_list, discarded_list, \
                excluded_list, discarded_list, optimal_dows, UE_list, E_list = tmp
            P0_list.append(dow_optimum)
    
    if P0_list:
        obj_vals = [dow.obj_value for dow in P0_list]
        idx_min = obj_vals.index(min(obj_vals))
        best_solution = P0_list[idx_min]
        return best_solution
    return None
                