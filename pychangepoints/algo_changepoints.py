import numpy as np
from .cython_pelt import cpelt, cbin_seg,  cseg_neigh
from .multiple_dim import cbin_seg_multiple, cpelt_multiple, cseg_neigh_multiple
from .nonparametric import cnp_pelt

from sklearn.decomposition import PCA
import pandas as pd
def multiple_preprocessing(df):
    data_scale = df.sub(df.mean(0).values, axis=1).div(df.std(0).values, axis=1)
    pca = PCA()
    pca.fit(data_scale)
    h = pca.transform(data_scale)
    pca_frame = pd.DataFrame(h)
    pca_frame_scale = pca_frame.sub(pca_frame.mean(0), axis=1).div(pca_frame.std(0), axis=1)
    return pca_frame_scale
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
    size_ts = len(data.index)
    stats_ts = np.zeros((size_ts+1,3))
    mean = np.mean(times_series)
    stats_ts[:,0] = np.append(0,times_series.cumsum())
    stats_ts[:,1] = np.append(0,(times_series**2).cumsum())
    stats_ts[:,2] = np.append(0,((times_series-mean)**2).cumsum())

    stats_ts_pelt = np.concatenate([stats_ts[:,0],stats_ts[:,1],stats_ts[:,2]])
    return  cpelt(stats_ts_pelt,pen_*np.log(size_ts),minseg,size_ts,method)

def np_pelt(data, pen_, minseg=10,nquantiles=10, method="mbic_nonparametric_ed"):
    times_series =data.values
    size_ts = times_series.shape[0]
    method_=method
    nquantiles_= int(2*np.log(size_ts))
    sumstat = nonpamametric_ed_sumstat(data,nquantiles_)
    return  cnp_pelt(sumstat, pen_,minseg,size_ts-1,nquantiles_,method_)
def segneigh( data, Q, method):
    times_series =data.values
    mean = np.mean(times_series)
    size_ts = times_series.shape[0]
    stats_ts = np.zeros((size_ts+1,3))
    stats_ts[:,0] = np.append(0,times_series.cumsum())
    stats_ts[:,1] = np.append(0,(times_series**2).cumsum())
    stats_ts[:,2] = np.append(0,((times_series-mean)**2).cumsum())

    return cseg_neigh( stats_ts, Q, method)
def binseg(data, Q, minseg, method):
    times_series =data.values
    mean = np.mean(times_series)
    size_ts = times_series.shape[0]
    stats_ts = np.zeros((size_ts+1,3))
    stats_ts[:,0] = np.append(0,times_series.cumsum())
    stats_ts[:,1] = np.append(0,(times_series**2).cumsum())
    stats_ts[:,2] = np.append(0,((times_series-mean)**2).cumsum())

    return cbin_seg( stats_ts, Q, minseg, method)
def binseg_multiple(data, Q, minseg, method):
    data_process = multiple_preprocessing(data)
    times_series =data_process.values
    mean = np.mean(times_series,axis=0)
    size_ts = times_series.shape[0]
    dim_ts = times_series.shape[1]
    stats_ts = np.zeros((size_ts+1,3,dim_ts))
    zeros = np.zeros(dim_ts)
    stats_ts[:,0,:] = np.insert(times_series.cumsum(axis=0), 0, zeros, axis=0)
    stats_ts[:,1,:] = np.insert((times_series**2).cumsum(axis=0), 0, zeros, axis=0)
    stats_ts[:,2,:] = np.insert(((times_series-mean)**2).cumsum(axis=0),0, zeros, axis=0)
    return cbin_seg_multiple( stats_ts, Q, minseg, method)
def pelt_multiple(data, pen_, minseg, method):
    data_process = multiple_preprocessing(data)
    times_series =data_process.values
    mean = np.mean(times_series,axis=0)
    size_ts = times_series.shape[0]
    dim_ts = times_series.shape[1]
    stats_ts = np.zeros((size_ts+1,3,dim_ts))
    zeros = np.zeros(dim_ts)
    stats_ts[:,0,:] = np.insert(times_series.cumsum(axis=0), 0, zeros, axis=0)
    stats_ts[:,1,:] = np.insert((times_series**2).cumsum(axis=0), 0, zeros, axis=0)
    stats_ts[:,2,:] = np.insert(((times_series-mean)**2).cumsum(axis=0),0, zeros, axis=0)
    return  cpelt_multiple(stats_ts, pen_*np.log(size_ts), minseg, size_ts, method)
def segneigh_multiple( data, Q, method):
    data_process = multiple_preprocessing(data)
    times_series =data_process.values
    mean = np.mean(times_series,axis=0)
    size_ts = times_series.shape[0]
    dim_ts = times_series.shape[1]
    stats_ts = np.zeros((size_ts+1,3,dim_ts))
    zeros = np.zeros(dim_ts)
    stats_ts[:,0,:] = np.insert(times_series.cumsum(axis=0), 0, zeros, axis=0)
    stats_ts[:,1,:] = np.insert((times_series**2).cumsum(axis=0), 0, zeros, axis=0)
    stats_ts[:,2,:] = np.insert(((times_series-mean)**2).cumsum(axis=0),0, zeros, axis=0)
    return cseg_neigh_multiple( stats_ts, Q, method)
