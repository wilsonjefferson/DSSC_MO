from src.utils.dow import DOW


def sort_by_topology(local_optimum:DOW, neighbours:list) -> list:
    diff_obj_vals = [dow.obj_value - local_optimum.obj_value for dow in neighbours]
    topology = [{'idx': i, 'val': diff} for i, diff in enumerate(diff_obj_vals)]

    def sort_by_value(d:dict):
        return d['val']

    topology.sort(key=sort_by_value)
    indexing = [item['idx'] for item in topology]
    topology = [neighbours[i] for i in indexing]

    return topology
