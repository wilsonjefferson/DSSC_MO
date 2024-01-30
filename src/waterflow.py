from collections import Counter

from src.larp import LARP
from src.utils.utils_waterflow.local_search import local_search, erosion

from src.utils.utils_waterflow.dow import DOW
from src.utils.utils_waterflow.clouds_generator import clouds_generator


def waterflow(larp:LARP, max_cloud:int, max_pop:int, max_UIE:int, min_ero:int) -> DOW:
    '''
        This function represents the WaterFlow Algorithm (WFA), a meta-heuristic algorithm
        used to find an "acceptable" solution in a "reasonable" amount of time. This
        implementation of the WFA try to find a local/global optimal solution for the
        Location-Assignement-Routing Problem, implemented by the LARP class.

        Arguments
        ---------
        larp:LARP
        Instance of the LARP model, already initialized (model build completed)

        max_cloud:int
        Integer number of clouds to generate

        max_pop:int
        Integer number of dows (drop-of-waters) each cloud has to generate

        max_UIE:int
        Integer number of maximum iterations to explore/exploit a position

        min_ero:int
        Integer number of minimum count of dows required to start erosion process

        Return
        ------
        best_solution:DOW
        Best dow (drop-of-water), aka solution
    '''
    optimal_dows = dict()
    P0_list = list()
    UE_list = list()
    E_list = list()

    excluded_list = list()
    discarded_list = list()

    # generator of clouds, generate at most max_cloud clouds
    clouds = clouds_generator(max_cloud, larp, max_pop)

    for cloud in clouds:
        # cloud generate a set of dow-of-water (dow)
        rainfall, discarded_dows = cloud.make_rain(E_list, discarded_list)
        discarded_list.extend(discarded_dows) # no feasible dows met during dows generation
        
        ### Exploration Phase ###
        for dow in rainfall:
            # gravity force push dow to a local optimal position (or solution)
            local_optimum, neighbours, excluded_dows, discarded_dows = local_search(larp, dow)
            excluded_list.extend(excluded_dows) # feasible dows excluded since less optimal than local optimum
            discarded_list.extend(discarded_dows) # no feasible position evaluated during local search
            UE_list.append(local_optimum) # for erosion process
        
            if local_optimum not in optimal_dows.keys():
                optimal_dows[local_optimum] = neighbours # store local optimal and his neighbour positions
        
        # erosion condition: a certain position is eligible for the erosion
        # process is a minimum number of min_ero dows converged to the same position
        dow_occurances = Counter(UE_list)
        dow_occurances = [dow for dow, occurs in dow_occurances.items() if occurs >= min_ero]
        for dow in dow_occurances:
            
            neighbours = optimal_dows[dow]

            # start erosion process for eligible dow
            tmp = erosion(larp, dow, neighbours, max_UIE,
                excluded_list, discarded_list, optimal_dows, 
                P0_list, UE_list, E_list)
            
            dow_optimum, _, excluded_list, discarded_list, \
                excluded_list, discarded_list, optimal_dows, UE_list, E_list = tmp
            P0_list.append(dow_optimum) # store (new) optimal position in P0
    
    if P0_list:
        # retrieve best position from P0
        obj_vals = [dow.obj_value for dow in P0_list]
        idx_min = obj_vals.index(min(obj_vals))
        best_solution = P0_list[idx_min]
        return best_solution
    return None
                