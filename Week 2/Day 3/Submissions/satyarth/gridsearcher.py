import joblib
from dask.distributed import Client, LocalCluster
from sklearn.model_selection import GridSearchCV

from config import n_folds

def searcher(estimator, param_grid, X_train, y_train, n_workers=4): # Launches distributed hyperparameter search, returns the best estimator found
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1, dashboard_address=None)
    client = Client(cluster)

    grid_search = GridSearchCV(estimator, param_grid, verbose=2, cv=n_folds, n_jobs=-1)

    with joblib.parallel_backend("dask", scatter=[X_train, y_train]):
        grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_