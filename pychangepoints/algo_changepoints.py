import numpy as np
from .cython_pelt import cpelt, cbin_seg
from .cython_pelt import cnp_pelt

def nonpamametric_ed_sumstat( data, K=10):
    ts_data = data.ix[:,0]
    n = len(ts_data)-1
    K= min( K,n)
    Q=np.zeros((K,n+1))
    x = ts_data.sort_values(ascending=True).values
    yK = -1 + ( 2*np.arange(1,K+1)/K-1/K)
    c= -np.log(2*n-1)
    pK = (1+np.exp(c*yK))**(-1)
    for i in range(1,K):
        j = int((n-1)*pK[i] + 1)
        Q[i,:]= np.cumsum(ts_data.values<x[j])+0.5*np.cumsum(ts_data.values==x[j])
    return Q
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

def np_pelt(data, pen_, minseg=10,nquantiles=10, method="nonparametric_ed"):
    times_series =data.values
    size_ts = times_series.shape[0]
    print(size_ts)
    method_=method
    nquantiles_= int(2*np.log(size_ts))
    sumstat = nonpamametric_ed_sumstat(data,nquantiles_)
    return  cnp_pelt(sumstat, pen_,minseg,size_ts-1,nquantiles_,method_)

def binseg(data, Q, minseg, method):
    times_series =data.values
    mean = np.mean(times_series)
    size_ts = times_series.shape[0]
    stats_ts = np.zeros((size_ts,3))
    stats_ts[:,0] = times_series.cumsum()
    stats_ts[:,1] = (times_series**2).cumsum()
    stats_ts[:,2] = ((times_series-mean)**2).cumsum()

    return cbin_seg( stats_ts, Q, minseg, method)
