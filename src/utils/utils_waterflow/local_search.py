from src.larp import LARP
from src.utils.utils_waterflow.dow import DOW
from src.utils.utils_waterflow.neighbourhood_strategies.opt_1_neighbourhood import opt_1
from src.utils.utils_waterflow.neighbourhood_strategies.swap_neighbourhood import swap
from src.utils.utils_waterflow.sort_topology import sort_by_topology


def local_search(larp:LARP, dow:DOW) -> tuple:
    '''
        Local search algorithm: starting from a given solution, the function
        apply in sequence the opt_1 and swap neighbourhood structures to find
        an improved solution, i.e. a new local optimum. 
        Apart for the improved solution, the function return also the:
        - neighbours, other solutions aorund the improved solution 
        - discarded dows, no feasible solutions;
        - excluded dows, feasible solutions but worse than the improved solution

        Arguments
        ---------
        larp:LARP
        An instance of the LARP model

        dow:DOW
        A drop-of-water (dow) representing a certain solution (a local optimum)

        Return
        ------
        local_optimum:dow
        A drop-of-water (dow) representing a local optimum

        neighbours:list
        Other solutions aorund the improved solution 

        discarded_dows:list
        List of no feasible solution evaluated using the LARP model

        excluded_dows:list
        List of feasible solutions (but worse than the improved solution) evaluated using LARP
    '''
    
    local_optimum = dow
    excluded_dows = list()
    discarded_dows = list()
    neighbours = None
    
    while True:

        solution, neighbours, no_feasible_dows = opt_1(larp, local_optimum)
        discarded_dows.extend(no_feasible_dows)

        # until an improved solution is found, continue the local search
        # using the opt_1 neighbourhood structure
        if solution != local_optimum:
            excluded_dows.extend([local_optimum])
            excluded_dows.extend(neighbours)
            local_optimum = solution
            continue
        
        # once no improved solution is found with the opt_1 neighbourhood structure,
        # continue the local search with the swap neighbourhood structure
        while True:
            solution, neighbours, no_feasible_dows = swap(larp, local_optimum)
            discarded_dows.extend(no_feasible_dows)

            # if no improved solution is found, stop the local search
            if solution == local_optimum:
                break

            excluded_dows.extend([local_optimum])
            excluded_dows.extend(neighbours)
            local_optimum = solution
        
        break
    
    return local_optimum, neighbours, excluded_dows, discarded_dows

def dow_seen(dow:DOW, excluded_list:list, discarded_list:list, UE_list:list, E_list:list) -> bool:
    '''
        Check if a certain drop-of-water (dow) was already evaluated.

        Arguments
        ---------
        dow:DOW
        A drop-of-water (dow) representing a certain solution

        excluded_list:list
        List of excluded dows, feasible solutions but worse than a discovered local optimum

        discarded_list:list
        List of discarded dows, no feasible solutions

        UE_list:list
        List of un-eroded "directions", i.e. dows whom the erosion process is not execute (yet)

        E_list:list
        List of eroded "directions", i.e. dows which are already used for the erision process

        Return
        ------
        bool
        True if dow was already seen, False otherwise
    '''

    return (dow in excluded_list or 
            dow in discarded_list or 
            dow in UE_list or 
            dow in E_list)

def get_next_neighbour(neighbours:list, excluded_list:list, 
                        discarded_list:list, UE_list:list, E_list:list) -> DOW:
    '''
        Retrieve the next "acceptable" neighbour from the list of ordered neighbours.

        Arguments
        ---------
        neighbours:list
        List of neighbours, i.e. dows

        excluded_list:list
        List of excluded dows, feasible solutions but worse than a discovered local optimum

        discarded_list:list
        List of discarded dows, no feasible solutions

        UE_list:list
        List of un-eroded "directions", i.e. dows whom the erosion process is not execute (yet)

        E_list:list
        List of eroded "directions", i.e. dows which are already used for the erision process

        Return
        ------
        neighbour:DOW or None
        return the next "acceptable" neighbour dow
    '''

    neighbour = None
    for dow in neighbours:
        if not dow_seen(dow, excluded_list, discarded_list, UE_list, E_list):
            neighbour = dow
            break

    return neighbour

def erosion(larp:LARP, local_optimum:DOW, neighbours:list, max_UIE:int,
            excluded_list:list, discarded_list:list, optimal_dows:dict, 
            P0_list:list, UE_list:list, E_list:list) -> tuple:
    '''
        Erosion process applied to a certain local optimum, this is the
        exploitation phase of the WaterFlow algorithm where the porpose
        is to to focus around a certain solution to see if it is possible
        to discover an improved solution.

        The topology parameter is computed for the current local optimum
        and it is used to set an ordering over the neighbour dows. Than,
        the erosion process excecute a local search over each neighbour
        until an improved solution is discovered, or no more dows are left.

        Arguments
        ---------
        larp:LARP
        An instance of the LARP model

        local_optimum:DOW
        A drop-of-water (dow) representing a local optimum solution

        neighbours:list
        List of neighbour dows, i.e. neighbour solutions of the local optimum

        max_UIE:int
        Integer number of maximum tentatives to execute the local search

        excluded_list:list
        List of excluded dows, feasible solutions but worse than a discovered local optimum

        discarded_list:list
        List of discarded dows, no feasible solutions

        optimal_dows:dict
        Dictionary representation of the discovered local optimum during the exploration phase

        P0_list:list
        List of all the best solution found until now

        UE_list:list
        List of un-eroded "directions", i.e. dows whom the erosion process is not execute (yet)

        E_list:list
        List of eroded "directions", i.e. dows which are already used for the erision process

        Return
        ------
        local_optimum:DOW
        New local optimum solution

        neighbours:list
        List of neighbours of the new local optimum

        excluded_list:list
        excluded_list updated
        
        discarded_list:list
        discarded_list updated
        
        optimal_dows:dict
        optimal_dows updated
        
        UE_list:list
        UE_list updated
        
        E_list:list
        E_list updated
        
    '''

    topology = sort_by_topology(local_optimum, neighbours)
    # print('neighbours topology:', topology)

    local_solution = None
    better_solution = False
    i = 0
    while i < len(topology) and not better_solution:
        tentative = 0
        curr_dow = topology[i]
        while tentative < max_UIE:
            local_solution, local_neighbours, excluded_dows, discarded_dows = local_search(larp, curr_dow)
            excluded_list.extend(excluded_dows)
            discarded_list.extend(discarded_dows)
            
            seen = dow_seen(local_solution, excluded_list, discarded_list, UE_list, E_list)
            # print('dow already seen?', seen)
            
            if not seen:

                if local_solution.obj_value < local_optimum.obj_value:
                    print('local solution is better than the local optimum!')
                    optimal_dows[local_solution] = local_neighbours
                    P0_list.append(local_solution)
                    UE_list.append(local_solution)
                    better_solution = True
                    break
                
                print('local solution is not better than the local optimum.')
                print('let\'s see if another tentative is possible searching from local solution...')
                
                curr_dow = local_solution

            elif local_solution == local_optimum:
                print('local solution is the local optimum.')
                print('search for the best neighbour of the local optimum...')
                curr_dow = get_next_neighbour(local_neighbours, excluded_list, 
                        discarded_list, UE_list, E_list)

                if curr_dow is None:
                    print('no aligible neighbour is found.')
                    local_solution = None
                    break

                print('aligible neighbour is found!')
                print('continue erosion process with eligible neighbour...')

            else:
                # local_solution was already seen and it is not the local_optimum.
                # So, it means local_solution belongs to one of the following list:
                # excluded_list, discarded_list or E_list. Therefore, the erosion
                # process is fully blocked!
                local_solution = None # reset local_solution
                tentative = max_UIE # stop while-loop
            tentative += 1
        i += 1

    if local_solution and better_solution:
        print('continue erosion process with local solution...')
        return erosion(larp, local_solution, local_neighbours, max_UIE,
                       excluded_list, discarded_list, optimal_dows, 
                       P0_list, UE_list, E_list)

    UE_list.remove(local_optimum)
    E_list.append(local_optimum)

    print('no better solution than the local optimum was found.')
    print('local optimum is completelly erored!')

    return (local_optimum, [], excluded_list, discarded_list, 
            optimal_dows, UE_list, E_list)
