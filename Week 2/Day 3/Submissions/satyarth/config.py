param_grid = {
    "max_depth": [2, 4],
    "gamma": [0.1, 1]
}


data_url = "https://storage.googleapis.com/dask-tutorial-data/nycflights.tar.gz"
data_dir = 'data'
subsample = 0.3

n_workers = 4
n_folds = 3
n_estimators = 69