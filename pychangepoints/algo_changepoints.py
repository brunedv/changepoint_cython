"""
Main functions of the package linked with cython
"""
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA


from .cython_pelt import cpelt, cbin_seg, cseg_neigh
from .multiple_dim import cbin_seg_multiple, cpelt_multiple, cseg_neigh_multiple
from .nonparametric import cnp_pelt


def multiple_preprocessing(data_frame):
    """
    Preprocessing of the multivariate data before segmentation
    Standardization with PCA
    """
    data_scale = data_frame.sub(data_frame.mean(0).values, axis=1)\
        .div(data_frame.std(0).values, axis=1)
    pca = PCA()
    pca.fit(data_scale)
    pca_scale = pca.transform(data_scale)
    pca_frame = pd.DataFrame(pca_scale)
    pca_frame_scale = pca_frame.sub(pca_frame.mean(0), axis=1).div(pca_frame.std(0), axis=1)
    return pca_frame_scale

def nonpamametric_ed_sumstat(data, n_stats=10):
    """
    Compute moments of the time series
    """
    ts_data = data.iloc[:, 0]
    size_data = len(ts_data)-1
    n_stats = min(n_stats, size_data)
    resultat = np.zeros((n_stats, size_data+1))
    ts_data_array = ts_data.sort_values(ascending=True).values
    y_k = -1 + (2*np.arange(1, n_stats+1)/n_stats-1/n_stats)
    log_size = -np.log(2*size_data-1)
    p_k = (1+np.exp(log_size*y_k))**(-1)
    for i in range(1, n_stats):
        j = int((size_data-1)*p_k[i] + 1)
        resultat[i, :] = np.cumsum(ts_data.values < ts_data_array[j])+\
            0.5*np.cumsum(ts_data.values == ts_data_array[j])
    return resultat

def pelt(data, pen_, minseg, method):
    """
    Univariate Pelt algorithm
    """
    times_series = data.values
    size_ts = len(data.index)
    stats_ts = np.zeros((size_ts+1, 3))
    mean = np.mean(times_series)
    stats_ts[:, 0] = np.append(0, times_series.cumsum())
    stats_ts[:, 1] = np.append(0, (times_series**2).cumsum())
    stats_ts[:, 2] = np.append(0, ((times_series-mean)**2).cumsum())

    stats_ts_pelt = np.concatenate([stats_ts[:, 0], stats_ts[:, 1], stats_ts[:, 2]])
    return  cpelt(stats_ts_pelt, pen_*np.log(size_ts), minseg, size_ts, method)

def np_pelt(data, pen_, minseg=10, nquantiles=10, method="mbic_nonparametric_ed"):
    """
    Univariate Pelt non parametric algorithm
    """
    times_series = data.values
    size_ts = times_series.shape[0]
    method_ = method
    nquantiles_ = int(2*np.log(size_ts))
    sumstat = nonpamametric_ed_sumstat(data, nquantiles_)
    return  cnp_pelt(sumstat, pen_, minseg, size_ts-1, nquantiles_, method_)

def segneigh(data, nb_cpts, method):
    """
    Univariate segmenttation neighboorhood algorithm
    """
    times_series = data.values
    mean = np.mean(times_series)
    size_ts = times_series.shape[0]
    stats_ts = np.zeros((size_ts+1, 3))
    stats_ts[:, 0] = np.append(0, times_series.cumsum())
    stats_ts[:, 1] = np.append(0, (times_series**2).cumsum())
    stats_ts[:, 2] = np.append(0, ((times_series-mean)**2).cumsum())

    return cseg_neigh(stats_ts, nb_cpts, method)

def binseg(data, nb_cpts, minseg, method):
    """
    Univariate binary segmentation
    """
    times_series = data.values
    mean = np.mean(times_series)
    size_ts = times_series.shape[0]
    stats_ts = np.zeros((size_ts+1, 3))
    stats_ts[:, 0] = np.append(0, times_series.cumsum())
    stats_ts[:, 1] = np.append(0, (times_series**2).cumsum())
    stats_ts[:, 2] = np.append(0, ((times_series-mean)**2).cumsum())

    return cbin_seg(stats_ts, nb_cpts, minseg, method)

def binseg_multiple(data, nb_cpts, minseg, method):
    """
    Multivariate binary segmentation
    """
    data_process = multiple_preprocessing(data)
    times_series = data_process.values
    mean = np.mean(times_series, axis=0)
    size_ts = times_series.shape[0]
    dim_ts = times_series.shape[1]
    stats_ts = np.zeros((size_ts+1, 3, dim_ts))
    zeros = np.zeros(dim_ts)
    stats_ts[:, 0, :] = np.insert(times_series.cumsum(axis=0), 0, zeros, axis=0)
    stats_ts[:, 1, :] = np.insert((times_series**2).cumsum(axis=0), 0, zeros, axis=0)
    stats_ts[:, 2, :] = np.insert(((times_series-mean)**2).cumsum(axis=0), 0, zeros, axis=0)
    return cbin_seg_multiple(stats_ts, nb_cpts, minseg, method)

def pelt_multiple(data, pen_, minseg, method):
    """
    Multivariate PELT segmentation algorithm
    """
    data_process = multiple_preprocessing(data)
    times_series = data_process.values
    mean = np.mean(times_series, axis=0)
    size_ts = times_series.shape[0]
    dim_ts = times_series.shape[1]
    stats_ts = np.zeros((size_ts+1, 3, dim_ts))
    zeros = np.zeros(dim_ts)
    stats_ts[:, 0, :] = np.insert(times_series.cumsum(axis=0), 0, zeros, axis=0)
    stats_ts[:, 1, :] = np.insert((times_series**2).cumsum(axis=0), 0, zeros, axis=0)
    stats_ts[:, 2, :] = np.insert(((times_series-mean)**2).cumsum(axis=0), 0, zeros, axis=0)
    return  cpelt_multiple(stats_ts, pen_*np.log(size_ts), minseg, size_ts, method)

def segneigh_multiple(data, nb_cpts, method):
    """
    Multivariate Segmentation neigborhhod algorithm
    """
    data_process = multiple_preprocessing(data)
    times_series = data_process.values
    mean = np.mean(times_series, axis=0)
    size_ts = times_series.shape[0]
    dim_ts = times_series.shape[1]
    stats_ts = np.zeros((size_ts+1, 3, dim_ts))
    zeros = np.zeros(dim_ts)
    stats_ts[:, 0, :] = np.insert(times_series.cumsum(axis=0), 0, zeros, axis=0)
    stats_ts[:, 1, :] = np.insert((times_series**2).cumsum(axis=0), 0, zeros, axis=0)
    stats_ts[:, 2, :] = np.insert(((times_series-mean)**2).cumsum(axis=0), 0, zeros, axis=0)
    return cseg_neigh_multiple(stats_ts, nb_cpts, method)