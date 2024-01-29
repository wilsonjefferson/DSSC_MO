from src.utils.get_data import random_data
from src.larp import LARP

from src.utils.clouds_generator import clouds_generator
from src.local_search import local_search


if __name__ == '__main__':

    k_vehicles = 5
    Q_vehicle_capacity = 2000
    facility = 'F'

    n_fields_instances = 100
    m_storages_instances = 20

    use = (n_fields_instances, m_storages_instances)
    fields, storages, households, f, q, fs_dist, cs_dist, d = random_data(*use)

    larp = LARP(facility, 
                k_vehicles, 
                Q_vehicle_capacity, 
                fields, 
                storages, 
                households, 
                f, 
                q, 
                fs_dist, 
                cs_dist, 
                d)
    
    larp.build()

    dows = dict()
    P0_list = list()
    UE_list = list()
    E_list = list()

    discarded_list = list()

    max_cloud = 10
    max_pop = 10

    clouds = clouds_generator(max_cloud, larp, max_pop)

    for cloud in clouds:
        rainfall, discarded_dows = cloud.make_rain(E_list, discarded_list)
        discarded_list.extend(discarded_dows)

        for dow in rainfall:
            local_optimum, neighbours, discarded_dows = local_search(larp, dow)
            dows[local_optimum] = neighbours
            discarded_list.extend(discarded_dows)
            