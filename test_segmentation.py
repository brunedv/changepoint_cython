import numpy as np 
import time 
import pandas as pd
from pychangepoints  import  cython_pelt, algo_changepoints
if __name__ == '__main__':
    size_ts = 5000
    cpts_true = [0,100,800,size_ts]
    nb_seg = len(cpts_true)-1
    nb_cpts = nb_seg-1
    mean=np.array([2,1,5])
    var=np.array([0.5,0.2,0.01])

    method = "mbic_meanvar"
    pen_ = 10.0
    minseg = 2
    time_series=np.zeros(size_ts)

    for j in range(0,nb_seg):
        time_series[cpts_true[j]:cpts_true[j+1]]=np.random.normal(mean[j],var[j],size=cpts_true[j+1]-cpts_true[j])

    stats_ts = np.zeros((size_ts,3))
    mean = np.mean(time_series)
    start = time.time()
    stats_ts[:,0] = time_series.cumsum()
    stats_ts[:,1] = (time_series**2).cumsum()
    stats_ts[:,2] = ((time_series-mean)**2).cumsum()
    stats_ts_pelt=np.concatenate([stats_ts[:,0],stats_ts[:,1],stats_ts[:,2]])
    print(time.time()-start)
    start = time.time()

    print("PELT",algo_changepoints.pelt(pd.DataFrame(time_series),pen_,minseg,method),time.time()-start)
    start = time.time()

    print("BinSeg",algo_changepoints.binseg(pd.DataFrame(time_series),10,2,method),time.time()-start)
    start = time.time()
    print("NPPELT",algo_changepoints.np_pelt(pd.DataFrame(time_series),pen_*np.log(size_ts),10),time.time()-start)
    start = time.time()
    print("Segneigh", algo_changepoints.segneigh(pd.DataFrame(time_series),3,method),time.time()-start)
