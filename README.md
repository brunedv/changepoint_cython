# changepoint_cython
A cython version of the changepoint and  changepoint.np R package, see https://github.com/cran/changepoint and https://github.com/cran/changepoint.np.

Implemented algorithms:
- Binary Segmentation, PELT, and Segmentation Neigborhood: monovariate and multivariate time series.
- Non parametric PELT: monovariate variate.

Model implemented for the monovariate case :
- Normal (mean, variance, or mean and variance).
- Poisson (mean and variance).
- Exponential (mean and variance).

Model implemented for the multivariate case :
- Normal (mean).


Working on Windows/Linux and python3.6
## Install
Download and run:
```shell
pip3 install .
```
## Tests 
Synthetic examples
```shell

python3 test_segmentation.py

python3 test_segmentation_multiple.py
```
## Test on well bore data 
See notebook "segmentation example on well log data.ipynb"
```python
import pandas as pd
import numpy as np
import lasio
from pychangepoints import cython_pelt, algo_changepoints
```
Loading and cleaning data
```python
well_log_las = lasio.read('./data/2120913D.las')
well_log_df = well_log_las.df()

list_var=['GR','NPOR','RHOZ','PEFZ']
well_log_data = well_log_df[list_var]

well_log_data = well_log_data[well_log_data>-1]
well_log_data.dropna(inplace=True)
well_log_data = well_log_data[well_log_data.index>6542]
```
Segmentation with PELT for multivariate time series.

```python
penalty_value = 5
min_segment_size = 20
model = 'mbic_mean'
list_cpts, nb_cpts = algo_changepoints.pelt_multiple(well_log_data,penalty_value, min_segment_size, model)
```
![Results of the segmentation](https://github.com/brunedv/changepoint_cython/blob/master/data/segmentation.png)
