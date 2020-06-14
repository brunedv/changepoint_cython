"""
Script testing the univariate segmentation
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
    PEN_ = 5
    MINSEG = 10
    TIME_SERIES = np.zeros(SIZE_TS)

    for j in range(0, NB_SEG):
        TIME_SERIES[CPTS_TRUE[j]:CPTS_TRUE[j+1]] = np.random.normal(MEAN_GEN[j], VAR_GEN[j], \
            size=CPTS_TRUE[j+1]-CPTS_TRUE[j])

    STATS_TS = np.zeros((SIZE_TS, 3))
    MEAN_TS = np.mean(TIME_SERIES)
    START = time.time()
    STATS_TS[:, 0] = TIME_SERIES.cumsum()
    STATS_TS[:, 1] = (TIME_SERIES**2).cumsum()
    STATS_TS[:, 2] = ((TIME_SERIES-MEAN_TS)**2).cumsum()
    STATS_TS_PELT = np.concatenate([STATS_TS[:, 0], STATS_TS[:, 1], STATS_TS[:, 2]])
    print(time.time()-START)
    START = time.time()

    print("PELT", algo_changepoints.pelt(pd.DataFrame(TIME_SERIES), PEN_, MINSEG, METHOD), \
         time.time()-START)
    START = time.time()
    print("BinSeg", algo_changepoints.binseg(pd.DataFrame(TIME_SERIES), 10, MINSEG, METHOD), \
         time.time()-START)
    START = time.time()
    print("NPPELT", algo_changepoints.np_pelt(pd.DataFrame(TIME_SERIES), PEN_*np.log(SIZE_TS), 10),\
         time.time()-START)
    START = time.time()
    print("Segneigh", algo_changepoints.segneigh(pd.DataFrame(TIME_SERIES), 10, METHOD), \
        time.time()-START)
