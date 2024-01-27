from src.larp import LARP
from src.get_data import random_data
from src.dows_generator import dows_generator
from src.opt_1_neighbourhood import opt_1
import os


if __name__ == '__main__':

    k_vehicles = 5
    Q_vehicle_capacity = 2000
    facility = 'F'

    n_fields_instances = 100
    m_storages_instances = 20

    folder_path = os.getcwd() + '\\backup\\waterflow\\'

    use = (n_fields_instances, m_storages_instances, folder_path)
    fields, storages, households, f, q, fs_dist, cs_dist, d = random_data(*use)

    larp_model = LARP(facility, 
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

    # NOTE: Since it is difficult to find an optimal solution 
    # trying with random X, Y and Z (decision variables)
    # we fix a random X and set guroby to find and stop the very first feasible solution
    larp_model.model.setParam('SolutionLimit', 1)
    larp_model.build()

    n_dows = 5
    generator = dows_generator(larp_model, n_dows, m_storages_instances, n_fields_instances, k_vehicles)
    rainfall = list()
    for i, dow in enumerate(generator):
        print('i:', i, 'objvalue:', dow.obj_value, sep=' ')
        rainfall.append(dow)

    for dow in rainfall:
        local_optimum = opt_1(larp_model, dow)
        print(local_optimum)