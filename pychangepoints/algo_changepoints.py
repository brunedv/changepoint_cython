import numpy as np
from .cython_pelt import cpelt, cbin_seg

def pelt(data, pen_, minseg, method):
    times_series =data.values
    size_ts = times_series.shape[0]
    print(size_ts)
    stats_ts = np.zeros((size_ts,3))
    mean = np.mean(times_series)
    stats_ts[:,0] = times_series.cumsum()
    stats_ts[:,1] = (times_series**2).cumsum()
    stats_ts[:,2] = ((times_series-mean)**2).cumsum()

    stats_ts_pelt = np.concatenate([stats_ts[:,0],stats_ts[:,1],stats_ts[:,2]])
    return  cpelt(stats_ts_pelt,pen_*np.log(size_ts),minseg,size_ts-1,method)

def binseg(data, Q, minseg, method):
    times_series =data.values
    mean = np.mean(times_series)
    size_ts = times_series.shape[0]
    stats_ts = np.zeros((size_ts,3))
    stats_ts[:,0] = times_series.cumsum()
    stats_ts[:,1] = (times_series**2).cumsum()
    stats_ts[:,2] = ((times_series-mean)**2).cumsum()

    return cbin_seg( stats_ts, Q, minseg, method)
