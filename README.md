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


Working on Windows/Linux and python3.x
## Install
###  PyPI
Source available on https://pypi.org/project/changepoint-cython/
```shell
pip install changepoint-cython
```
### From source
Download and run:
```shell
pip3 install -r requirements.txt
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

## Documentation

### PELT
Inputs:
- data: data as pandas dataframe,
- penalty_value: double, penalty value, (BIC penalty),
- min_segment_size: int, minimum segment size,
- model: str, statistical model (see cost_function.pxd).

Outputs:
- list_cpts: array of the postion of the change-points,
- nb_cpts: number of chnagepoint.

Usage:
```python
from pychangepoints import algo_changepoints

penalty_value = 5
min_segment_size = 20
model = 'mbic_mean'
list_cpts, nb_cpts = algo_changepoints.pelt(data, penalty_value, min_segment_size, model)
```
For the multivariate case, call pelt_multiple:
```python
from pychangepoints import algo_changepoints

penalty_value = 5
min_segment_size = 20
model = 'mbic_mean'
list_cpts, nb_cpts = algo_changepoints.pelt_multiple(data, penalty_value, min_segment_size, model)
```
### NP PELT
Inputs:
- data: as pandas dataframe,
- penalty_value: double, penalty value, (BIC penalty),
- min_segment_size: int, minimum segment size,
- nquantiles: int, number of quantiles,
- method: str, cost functioon (optional, only one implemented).

Outputs:
- list_cpts, array of the postion of the change-points
- nb_cpts, number of chnagepoint.

Usage:
```python
from pychangepoints import algo_changepoints

penalty_value = 5
min_segment_size = 20
model = 'mbic_nonparametric_ed'
nquantiles = 10
list_cpts, nb_cpts = algo_changepoints.np_pelt(data, penalty_value, min_segment_size, nquantiles, method = model)
```
### Binary Segmentation
Inputs:
- data: pandas dataframe,
- Q: int, number of changepoints,
- min_segment_size: int, minimum segment size,
- statistical model, str (see cost_function.pxd).

Output:
- array of the postion of the change-points,

Usage:
```python
from pychangepoints import algo_changepoints
Q = 5
min_segment_size = 20
model = 'mbic_mean'
list_cpts = algo_changepoints.binseg(data, Q, minseg, method)
```
For the multivariate case, call binseg_multiple:
```python
from pychangepoints import algo_changepoints
Q = 5
min_segment_size = 20
model = 'mbic_mean'
list_cpts = algo_changepoints.binseg_multiple(data, Q, minseg, method)
```
### Segmentation neighborhood
Inputs:
- data: pandas dataframe,
- Q: int, number of changepoints,
- statistical model, str (see cost_function.pxd).

Output:
- array of the postion of the change-points,

Usage:
```python
from pychangepoints import algo_changepoints
Q = 5
model = 'mbic_mean'
list_cpts = algo_changepoints.segneigh(data, Q, method)
```
For the multivariate case, call segneigh_multiple:
```python
from pychangepoints import algo_changepoints
Q = 5
model = 'mbic_mean'
list_cpts = algo_changepoints.segneigh_multiple(data, Q, method)
```
