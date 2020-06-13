import pytest
import numpy as np 
import pandas as pd
from pychangepoints  import   algo_changepoints


size_ts = 1000
cpts_true = [0, 400, 800, size_ts]
nb_seg = len(cpts_true)-1
nb_cpts = nb_seg-1
mean = np.array([10, 1, 5])
var = np.array([0.5, 0.5, 0.5])

method = "mbic_mean"
pen_ = 5
minseg = 10
time_series = np.zeros(size_ts)

for j in range(0, nb_seg):
	time_series[cpts_true[j]:cpts_true[j+1]] = np.random.normal(mean[j], var[j], size=cpts_true[j+1]-cpts_true[j])

stats_ts = np.zeros((size_ts, 3))
mean = np.mean(time_series)
stats_ts[:, 0] = time_series.cumsum()
stats_ts[:, 1] = (time_series**2).cumsum()
stats_ts[:, 2] = ((time_series-mean)**2).cumsum()
stats_ts_pelt = np.concatenate([stats_ts[:, 0], stats_ts[:, 1], stats_ts[:, 2]])

def test_pelt():
	res_seg = np.sort(algo_changepoints.pelt(pd.DataFrame(time_series), pen_, minseg, method)[0])
	
	assert res_seg[0] == cpts_true[1],"test failed"
	assert res_seg[1] == cpts_true[2],"test failed"
