from src.utils.utils_waterflow.dow import DOW


def sort_by_topology(local_optimum:DOW, neighbours:list) -> list:
    '''
        Compute the topology parameter for the given local optimum; a
        topology parameters is represented by a list of differences between
        the local optimum objective function and the objective function of 
        each neighbours solution (other dows).

        Arguments
        ---------
        local_optimum:DOW
        A drop-of-water (dow) representing a local optimum position

        neighbours:list
        List of neighbours solution of the local optimum

        Return
        ------
        topology:list
        Topology parameter for the local optimum
    '''

    diff_obj_vals = [dow.obj_value - local_optimum.obj_value for dow in neighbours]
    topology = [{'idx': i, 'val': diff} for i, diff in enumerate(diff_obj_vals)]

    def sort_by_value(d:dict):
        return d['val']

    topology.sort(key=sort_by_value)
    indexing = [item['idx'] for item in topology]
    topology = [neighbours[i] for i in indexing]

    return topology
