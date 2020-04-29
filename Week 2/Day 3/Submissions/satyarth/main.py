from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from dask_ml.xgboost import XGBRegressor as dXGBRegressor
from pandas import DataFrame
from time import time

from dataset import loader
from gridsearcher import searcher

from config import param_grid, n_workers, n_estimators # All configs loaded from a python file (felt JSON/command line
													   # args would be overkill, kept it simple)
def benchmark(estimator, param_grid, X, y): # Launches benchmarking given an estimator, dictionary
											# of hyperparameter values, and the input and target of the dataset
	X_train, X_test, y_train, y_test = train_test_split(X, y)
	start_time = time()
	best_estimator = searcher(xgb, param_grid, X_train, y_train, n_workers=n_workers)
	runtime = time() - start_time 
	preds = best_estimator.predict(X_test)
	mse = mean_squared_error(y_test, preds)
	return runtime, mse

def write_results(results): # Writes benchmarking results to a CSV file (pandas for this is overkill,
	df = DataFrame(results) # but I'm lazy
	df.to_csv("results.csv")

if __name__ == "__main__":
	print("Loading data...")
	X, y = loader()
	# del X, y
	print("Data loaded")


	xgb = XGBRegressor(n_estimators=n_estimators)
	dxgb = dXGBRegressor(n_estimators=n_estimators)

	descriptions = ["XGBoost", "XGBoost (dask-ml)"]
	results = []
	for estimator, description in zip([xgb, dxgb], descriptions):
		runtime, mse = benchmark(dxgb, param_grid, X, y)
		print("Model: {}".format(description))
		print("Runtime: {} s".format(runtime))
		print("MSE: {}".format(mse))
		results.append((description, runtime, mse))

	write_results(results)
	print("Results saved to `results.csv`")

	
