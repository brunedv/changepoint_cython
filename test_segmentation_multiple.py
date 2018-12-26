import numpy as np 
import time 
import pandas as pd
from pychangepoints  import  cython_pelt, algo_changepoints
if __name__ == '__main__':
    size_ts = 20000
    dim = 1
    cpts_true = [0,100,800,size_ts]
    nb_seg = len(cpts_true)-1
    nb_cpts = nb_seg-1
    mean=np.array([2,1,5])
    var=np.array([0.5,0.5,0.5])

    method = "mll_mean"
    pen_ = 10.0
    minseg = 2
    time_series=np.zeros((size_ts,3))
    for j in range(0,nb_seg):
            time_series[cpts_true[j]:cpts_true[j+1],:]=np.random.normal(mean[j],var[j],size=(cpts_true[j+1]-cpts_true[j],dim))

    start = time.time()
    data = pd.DataFrame(time_series)
    data_0 = pd.DataFrame(time_series[:,0])

    print( algo_changepoints.binseg_multiple(data,10,2,method),time.time()-start)
    print( algo_changepoints.pelt_multiple(data,1,10,method),time.time()-start)
    print( algo_changepoints.pelt(data_0,1,10,method),time.time()-start)

