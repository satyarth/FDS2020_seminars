from __future__ import print_function

import os
import numpy as np
import pandas as pd
import tarfile
import urllib.request
import zipfile
from glob import glob
import argparse

from config import data_url, data_dir

def download_flight_data(url, data_dir="data", n_rows=None):
    flights_raw = os.path.join(data_dir, 'nycflights.tar.gz')
    flightdir = os.path.join(data_dir, 'nycflights')
    jsondir = os.path.join(data_dir, 'flightjson')

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    if not os.path.exists(flights_raw):
        print("- Downloading NYC Flights dataset... ", end='', flush=True)
        urllib.request.urlretrieve(url, flights_raw)
        print("done", flush=True)

    if not os.path.exists(flightdir):
        print("- Extracting flight data... ", end='', flush=True)
        tar_path = os.path.join('data', 'nycflights.tar.gz')
        with tarfile.open(tar_path, mode='r:gz') as flights:
            flights.extractall('data/')
        print("done", flush=True)

    # Note: I opted not to bother with JSON here

    print("** Finished! **")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
   
    parser.add_argument('-u', '--url',
        default=data_url,
        help="Dataset source URL")
    parser.add_argument('-d', '--datadir', default=data_dir, 
        help="output directory")
    parser.add_argument('-n', '--nrows',default=None,
        help="Number of rows to keep")

    args = parser.parse_args()
    download_flight_data(url=args.url, data_dir=args.datadir, n_rows=args.nrows)

