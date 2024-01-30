from src.larp import LARP
from src.utils.dow import DOW
# from src.utils.opt_1_neighbourhood import opt_1
from src.utils.swap_neighbourhood import swap
from src.utils.sort_topology import sort_by_topology


def local_search(larp:LARP, dow:DOW) -> tuple:
    
    local_optimum = dow
    excluded_dows = list()
    discarded_dows = list()
    neighbours = None
    
    while True:

        # solution, neighbours, no_feasible_dows = opt_1(larp, local_optimum)
        solution, neighbours, no_feasible_dows = swap(larp, local_optimum)
        discarded_dows.extend(no_feasible_dows)

        # print('local_search | solution in loop:', solution)
        # print('local_search | local_optimum in loop:', local_optimum)

        if solution == local_optimum:
            break
        
        excluded_dows.extend([local_optimum])
        excluded_dows.extend(neighbours)
        local_optimum = solution
    
    return local_optimum, neighbours, excluded_dows, discarded_dows

def dow_seen(dow:DOW, excluded_list:list, discarded_list:list, UE_list:list, E_list:list) -> bool:
    return (dow in excluded_list or 
            dow in discarded_list or 
            dow in UE_list or 
            dow in E_list)

def get_next_neighbour(neighbours:list, excluded_list:list, 
                        discarded_list:list, UE_list:list, E_list:list) -> DOW:
    neighbour = None
    for dow in neighbours:
        if not dow_seen(dow, excluded_list, discarded_list, UE_list, E_list):
            neighbour = dow
            break

    return neighbour

def erosion(larp:LARP, local_optimum:DOW, neighbours:list, max_UIE:int,
            excluded_list:list, discarded_list:list, optimal_dows:dict, 
            P0_list:list, UE_list:list, E_list:list) -> tuple:

    topology = sort_by_topology(local_optimum, neighbours)
    # print('neighbours topology:', topology)

    local_solution = None
    for dow in topology:
        tentative = 0
        curr_dow = dow
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
                    UE_list.remove(local_optimum)
                    E_list.append(local_optimum)
                    break
                
                curr_dow = local_solution

            elif local_solution == local_optimum:
                print('local solution is the local optimum.')
                print('search for the best neighbour of the local optimum...')
                curr_dow = get_next_neighbour(local_neighbours, excluded_list, 
                        discarded_list, UE_list, E_list)

                if curr_dow:
                    print('aligible neighbour is found!')
                    print('continue erosion process with eligible neighbour...')
                else:
                    print('no aligible neighbour is found.')
                    local_solution = None
                    break
            else:
                # local_solution was already seen and it is not the local_optimum.
                # So, it means local_solution belongs to one of the following list:
                # excluded_list, discarded_list or E_list. Therefore, the erosion
                # process is fully blocked!
                local_solution = None # reset local_solution
                tentative = max_UIE # stop while-loop
            tentative += 1
    
    if local_solution:
        print('continue erosion process with local solution...')
        return erosion(larp, local_solution, local_neighbours, max_UIE,
                       excluded_list, discarded_list, optimal_dows, 
                       P0_list, UE_list, E_list)
    
    UE_list.remove(local_optimum)
    E_list.append(local_optimum)

    print('no better solution than the local optimum was found.')
    print('local optimum is completelly erored!')

    return (local_optimum, [], excluded_list, discarded_list, 
            excluded_list, discarded_list, optimal_dows, UE_list, E_list)
