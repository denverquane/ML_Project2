from scipy import sparse
import numpy as np
import tables as tb 
import pandas as pd
import csv 

cols = 61190
rows = 12000

# https://stackoverflow.com/a/27203245

filename = "training.h5"

df = pd.DataFrame.from_csv("training.csv", header=None, index_col=None)
# makes a dataframe from the csv data (should make it easier for some ops.)

df.to_hdf(filename, 'data', mode = 'w')
# converts the entire dataset to an hd5 file 

print(pd.read_hdf(filename, 'data'))