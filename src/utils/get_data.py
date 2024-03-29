import pandas as pd
import numpy as np


def random_data(n_fields:int, m_storages:int) -> tuple:
    '''
        Given number of fields and number of storages, this function
        generate random data in order to execute the LARP model for
        different size of the problem.

        Arguments
        ---------

        n_fields:int
        Integer value, it represents the number of fields considered

        m_storages:int
        Integer value, it represents the number of storaged considered

        Return
        ------

        tuple
        Tuple with fields, storages, households, f, q, fs_dist, cs_dist, pivot_d

    '''
    
    # construct storage and fields labels
    storages = ['S'+str(i) for i in range(m_storages)]
    fields = ['C'+str(j) for j in range(n_fields)]
    households = ['H'+str(h) for h in range(n_fields)] 
    
    rng = np.random.default_rng(None) # random generator

    # generate fields <-> storages distance dataframe
    # values randomly generated in the interval [1, 20]
    cs_dist = rng.integers(low=1, high=20, size=(n_fields, m_storages))
    cs_dist = pd.DataFrame(cs_dist, columns=storages, index=fields)

    # generate storages <-> storages (and facility) distance dataframe
    # values randomly generated in the interval [1, 20]
    fs_dist = rng.integers(low=1, high=20, size=(m_storages+1, m_storages+1))
    fs_dist = (fs_dist + fs_dist.T)/2
    fs_dist = pd.DataFrame(fs_dist, columns=storages+['F'], index=storages+['F'])

    # generate storage cost and capacity dataframe 
    # values randomly generated in the interval [50, 200]
    # storage capacity fixed at 1000 tons
    storage_cost = rng.integers(low=50, high=200, size=(m_storages,)).tolist()
    storage_capacity = np.full((m_storages,), 1000).tolist()
    
    f = dict(zip(storages, storage_cost))
    q = dict(zip(storages, storage_capacity))

    # generate dataframe representing the amount of agricultural waste
    # values randomly generated in the interval [1, 10]
    tmp_data = rng.integers(low=1, high=10, size=(n_fields, 1))
    pivot_d = np.concatenate([np.array(fields).reshape((n_fields, 1)), 
                            np.array(households).reshape((n_fields, 1)), 
                            tmp_data], axis=1)
    
    pivot_d = pd.DataFrame(pivot_d, columns=['cluster', 'household', 'demand'])
    pivot_d = pivot_d.pivot(index='cluster', columns='household', values='demand')
    pivot_d.fillna(0, inplace=True)
    pivot_d = pivot_d[households]
    pivot_d = pivot_d.reindex(fields)

    return fields, storages, households, f, q, fs_dist, cs_dist, pivot_d
