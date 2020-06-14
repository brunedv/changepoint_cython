"""
Script testing changepoints in multivariate time series
"""
import time

import numpy as np
import pandas as pd

from pychangepoints import algo_changepoints

if __name__ == '__main__':
    SIZE_TS = 1000
    CPTS_TRUE = [0, 400, 800, SIZE_TS]
    NB_SEG = len(CPTS_TRUE)-1
    NB_CPTS = NB_SEG-1
    MEAN_GEN = np.array([10, 1, 5])
    VAR_GEN = np.array([0.5, 0.5, 0.5])
    METHOD = "mbic_mean"
    PEN_ = 10
    MINSEG = 10
    TIME_SERIES = np.zeros((SIZE_TS, 3))
    DIM = 1

    for j in range(0, NB_SEG):
        TIME_SERIES[CPTS_TRUE[j]:CPTS_TRUE[j+1], :] = np.random.normal(MEAN_GEN[j], VAR_GEN[j], \
                size=(CPTS_TRUE[j+1]-CPTS_TRUE[j], DIM))

    START = time.time()
    DATA = pd.DataFrame(TIME_SERIES)
    DATA_0 = pd.DataFrame(TIME_SERIES[:, 0])

    print("BinSeg", algo_changepoints.binseg_multiple(DATA, 10, MINSEG, METHOD), time.time()-START)
    #print("PELT",algo_changepoints.pelt_multiple(data, pen_, minseg, method), time.time()-start)
    #print("SegNeigh",algo_changepoints.segneigh_multiple(data, 10, method), time.time()-start)
    #print( algo_changepoints.pelt(data_0,1,10,method),time.time()-start)

