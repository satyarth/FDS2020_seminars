import dask.dataframe as dd
from dask_ml.preprocessing import LabelEncoder
from os import listdir, path
import pandas as pd

from config import data_dir, subsample

def loader(): # Reads dataset path from config and returns X, y arrays
	df = dd.read_csv(path.join(data_dir, 'nycflights', '*.csv'),
                 dtype={'TailNum': str,
                        'CRSElapsedTime': float,
                        'Cancelled': bool},)
	df = df.drop(['AirTime', 'TailNum', 'TaxiIn', 'TaxiOut'], axis=1) # Drop columns with laaarge numbers of missing values
	df = df[~df.DepDelay.isna()] # Drop rows where DepDelay is not defined
	df = df.fillna(0)
	df = df.sample(frac=subsample)
	df_pd = df.compute()
	le = LabelEncoder()
	for col in ['UniqueCarrier', 'Origin', 'Dest']:
	    df_pd[col] = le.fit_transform(df_pd[col])

	X = df_pd.drop('DepDelay', axis=1).values
	y = df_pd.DepDelay.values
	# del df
	return X, y