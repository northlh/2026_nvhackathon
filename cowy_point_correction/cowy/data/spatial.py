
import numpy as np
import copy

def spatial_blocking(cowy, block_size, n_folds):
    lat, lon = cowy.lat_obs, cowy.lon_obs
    lat0, lon0 = np.floor(lat.min()), np.floor(lon.min())

    bx = np.floor((lon - lon0) / block_size)
    by = np.floor((lat - lat0) / block_size)
    folds = (bx + by) % n_folds

    train_idx = np.where(folds != 0)[0]
    test_idx = np.where(folds == 0)[0]

    train = copy.copy(cowy)
    test  = copy.copy(cowy)

    train.obs_lookup = cowy.obs_lookup[cowy.obs_lookup.idx_obs.isin(train_idx)].reset_index(drop=True)
    test.obs_lookup  = cowy.obs_lookup[cowy.obs_lookup.idx_obs.isin(test_idx)].reset_index(drop=True)

    return train, test
