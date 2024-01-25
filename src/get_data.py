import pandas as pd
import numpy as np
import os
import pickle

SEED = 66


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")

def random_data(n_fields:int, m_storages:int) -> tuple:

    folder_path = os.getcwd() + f"\\DSSC_MO\\backup\\"
    create_folder_if_not_exists(folder_path)

    storages = ['S'+str(i) for i in range(m_storages)]
    fields = ['C'+str(j) for j in range(n_fields)]
    households = ['H'+str(h) for h in range(n_fields)] 
    
    rng = np.random.default_rng(SEED)

    if not os.path.exists(folder_path + 'cs_dist.pkl'):
        cs_dist = rng.integers(low=1, high=20, size=(n_fields, m_storages))
        cs_dist = pd.DataFrame(cs_dist, columns=storages, index=fields)
        cs_dist.to_pickle(folder_path + 'cs_dist.pkl')
    else:
        cs_dist = pd.read_pickle(folder_path + 'cs_dist.pkl')
    
    if not os.path.exists(folder_path + 'fs_dist.pkl'):
        fs_dist = rng.integers(low=1, high=20, size=(m_storages+1, m_storages+1))
        fs_dist = (fs_dist + fs_dist.T)/2
        fs_dist = pd.DataFrame(fs_dist, columns=storages+['F'], index=storages+['F'])
        fs_dist.to_pickle(folder_path + 'fs_dist.pkl')
    else:
        fs_dist = pd.read_pickle(folder_path + 'fs_dist.pkl')
    
    if not os.path.exists(folder_path + 'f.pkl') or not os.path.exists(folder_path + 'q.pkl'):
        storage_cost = rng.integers(low=50, high=200, size=(m_storages,)).tolist()
        storage_capacity = np.full((m_storages,), 1000).tolist()
        
        f = dict(zip(storages, storage_cost))
        q = dict(zip(storages, storage_capacity))

        with open(folder_path+'f.pkl', 'wb') as file:
            pickle.dump(f, file)

        with open(folder_path+'q.pkl', 'wb') as file:
            pickle.dump(q, file)
    else:
        with open(folder_path+'f.pkl', 'rb') as file:
            f = pickle.load(file)

        with open(folder_path+'q.pkl', 'rb') as file:
            q = pickle.load(file)

    if not os.path.exists(folder_path + 'pivot_d.pkl'):
        tmp_data = rng.integers(low=1, high=10, size=(n_fields, 1))
        pivot_d = np.concatenate([np.array(fields).reshape((n_fields, 1)), 
                                np.array(households).reshape((n_fields, 1)), 
                                tmp_data], axis=1)
        
        pivot_d = pd.DataFrame(pivot_d, columns=['cluster', 'household', 'demand'])
        pivot_d = pivot_d.pivot(index='cluster', columns='household', values='demand')
        pivot_d.fillna(0, inplace=True)
        pivot_d = pivot_d[households]
        pivot_d = pivot_d.reindex(fields)
        pivot_d.to_pickle(folder_path + 'pivot_d.pkl')
    else:
        pivot_d = pd.read_pickle(folder_path + 'pivot_d.pkl')

    return fields, storages, households, f, q, fs_dist, cs_dist, pivot_d
