# changepoint_cython
A cython version of the changepoint and  changepoint.np R package, see https://github.com/cran/changepoint and https://github.com/cran/changepoint.np.

Implemented algorithms:  Binary Segmentation, PELT, Non parametric PELT, Segmentation Neigborhood.

Working on Windows/Linux and python3.6
## install
pip3 install -e .
## Test file
python3 test_segmentation.py
python3 test_segmentation_multiple.py
## Test on well bore data see (notebook)
```
import pandas as pd
import numpy as np
import lasio
from pychangepoints import cython_pelt, algo_changepoints
```
Loading and cleaning data
```
well_log_las = lasio.read('./data/2120913D.las')
well_log_df = well_log_las.df()

list_var=['GR','NPOR','RHOZ','PEFZ']
well_log_data = well_log_df[list_var]

well_log_data = well_log_data[well_log_data>-1]
well_log_data.dropna(inplace=True)
well_log_data = well_log_data[well_log_data.index>6542]
```
