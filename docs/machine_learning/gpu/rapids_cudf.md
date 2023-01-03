# RAPIDS cuDF

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ritchieng/deep-learning-wizard/blob/master/docs/machine_learning/gpu/rapids_cudf.ipynb)

## Environment Setup

### Check Version

#### Python Version


```python
# Check Python Version
!python --version
```

#### Ubuntu Version


```python
# Check Ubuntu Version
!lsb_release -a
```

    No LSB modules are available.
    Distributor ID:	Ubuntu
    Description:	Ubuntu 18.04.2 LTS
    Release:	18.04
    Codename:	bionic


#### Check CUDA Version


```python
# Check CUDA/cuDNN Version
!nvcc -V && which nvcc
```

    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2018 NVIDIA Corporation
    Built on Sat_Aug_25_21:08:01_CDT_2018
    Cuda compilation tools, release 10.0, V10.0.130
    /usr/local/cuda/bin/nvcc


#### Check GPU Version


```python
# Check GPU
!nvidia-smi
```

    Mon May 13 09:31:40 2019       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 418.56       Driver Version: 410.79       CUDA Version: 10.0     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
    | N/A   67C    P8    17W /  70W |      0MiB / 15079MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+


#### Setup:
Set up script installs
1. Updates gcc in Colab
1. Installs Conda
1. Install RAPIDS' current stable version of its libraries, as well as some external libraries including:
  1. cuDF
  1. cuML
  1. cuGraph
  1. cuSpatial
  1. cuSignal
  1. BlazingSQL
  1. xgboost
1. Copy RAPIDS .so files into current working directory, a neccessary workaround for RAPIDS+Colab integration.



```python
# This get the RAPIDS-Colab install files and test check your GPU.  Run this and the next cell only.
# Please read the output of this cell.  If your Colab Instance is not RAPIDS compatible, it will warn you and give you remediation steps.
!git clone https://github.com/rapidsai/rapidsai-csp-utils.git
!python rapidsai-csp-utils/colab/env-check.py
```


```python
# This will update the Colab environment and restart the kernel.  Don't run the next cell until you see the session crash.
!bash rapidsai-csp-utils/colab/update_gcc.sh
import os
os._exit(00)
```


```python
# This will install CondaColab.  This will restart your kernel one last time.  Run this cell by itself and only run the next cell once you see the session crash.
import condacolab
condacolab.install()
```


```python
# you can now run the rest of the cells as normal
import condacolab
condacolab.check()
```

### Installation of RAPIDS (including cuDF/cuML)
Many thanks to NVIDIA team for this snippet of code to automatically set up everything.


```python
# Installing RAPIDS is now 'python rapidsai-csp-utils/colab/install_rapids.py <release> <packages>'
# The <release> options are 'stable' and 'nightly'.  Leaving it blank or adding any other words will default to stable.
# The <packages> option are default blank or 'core'.  By default, we install RAPIDSAI and BlazingSQL.  The 'core' option will install only RAPIDSAI and not include BlazingSQL, 
!python rapidsai-csp-utils/colab/install_rapids.py stable
import os
os.environ['NUMBAPRO_NVVM'] = '/usr/local/cuda/nvvm/lib64/libnvvm.so'
os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/local/cuda/nvvm/libdevice/'
os.environ['CONDA_PREFIX'] = '/usr/local'
```

## Critical Imports


```python
# Critical imports
import nvstrings, nvcategory, cudf
import cuml
import os
import numpy as np
import pandas as pd
```

## Creating

### Create a Series of integers


```python
gdf = cudf.Series([1, 2, 3, 4, 5, 6])
print(gdf)
print(type(gdf))
```

    0    1
    1    2
    2    3
    3    4
    4    5
    5    6
    dtype: int64
    <class 'cudf.dataframe.series.Series'>


### Create a Series of floats


```python
gdf = cudf.Series([1., 2., 3., 4., 5., 6.])
print(gdf)
```

    0    1.0
    1    2.0
    2    3.0
    3    4.0
    4    5.0
    5    6.0
    dtype: float64


### Create a  Series of strings



```python
gdf = cudf.Series(['a', 'b', 'c'])
print(gdf)
```

    0    a
    1    b
    2    c
    dtype: object


### Create 3 column DataFrame
- Consisting of dates, integers and floats


```python
# Import
import datetime as dt

# Using a list of tuples
# Each element in the list represents a category
# The first element of the tuple is the category's name
# The second element of the tuple is a list of the values in that category
gdf = cudf.DataFrame([
    # Create 10 busindates ess from 1st January 2019 via pandas
    ('dates', pd.date_range('1/1/2019', periods=10, freq='B')),
    # Integers
    ('integers', [i for i in range(10)]),
    # Floats
    ('floats', [float(i) for i in range(10)])
])

# Print dataframe
print(gdf)
```

                         dates  integers  floats
    0 2019-01-01T00:00:00.000         0     0.0
    1 2019-01-02T00:00:00.000         1     1.0
    2 2019-01-03T00:00:00.000         2     2.0
    3 2019-01-04T00:00:00.000         3     3.0
    4 2019-01-07T00:00:00.000         4     4.0
    5 2019-01-08T00:00:00.000         5     5.0
    6 2019-01-09T00:00:00.000         6     6.0
    7 2019-01-10T00:00:00.000         7     7.0
    8 2019-01-11T00:00:00.000         8     8.0
    9 2019-01-14T00:00:00.000         9     9.0


### Create 2 column Dataframe
- Consisting of integers and string category


```python
# Using a list of tuples
# Each element in the list represents a category
# The first element of the tuple is the category's name
# The second element of the tuple is a list of the values in that category
gdf = cudf.DataFrame([
    ('integers', [1 ,2, 3, 4]),
    ('string', ['a', 'b', 'c', 'd'])
])

print(gdf)
```

       integers  string
    0         1       a
    1         2       b
    2         3       c
    3         4       d


### Create a 2 Column  Dataframe with Pandas Bridge
- Consisting of integers and string category
- For all string columns, you must convert them to type `category` for filtering functions to work intuitively (for now)


```python
# Create pandas dataframe
pandas_df = pd.DataFrame({
    'integers': [1, 2, 3, 4], 
    'strings': ['a', 'b', 'c', 'd']
})

# Convert string column to category format
pandas_df['strings'] = pandas_df['strings'].astype('category')

# Bridge from pandas to cudf
gdf = cudf.DataFrame.from_pandas(pandas_df)

# Print dataframe
print(gdf)
```

       integers  strings
    0         1        a
    1         2        b
    2         3        c
    3         4        d


## Viewing

### Printing Column Names


```python
gdf.columns
```




    Index(['integers', 'strings'], dtype='object')



### Viewing Top of DataFrame


```python
num_of_rows_to_view = 2 
print(gdf.head(num_of_rows_to_view))
```

       integers  strings
    0         1        a
    1         2        b


### Viewing Bottom of DataFrame


```python
num_of_rows_to_view = 3 
print(gdf.tail(num_of_rows_to_view))
```

       integers  strings
    1         2        b
    2         3        c
    3         4        d


## Filtering

### Method 1: Query

#### Filtering Integers/Floats by Column Values
- This only works for floats and integers, not for strings


```python
print(gdf.query('integers == 1'))
```

       integers  strings
    0         1        a


#### Filtering Strings by Column Values
- This only works for floats and integers, not for strings so this will return an error!


```python
print(gdf.query('strings == a'))
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-33-5cfd0345d51c> in <module>()
    ----> 1 print(gdf.query('strings == a'))
    

    /usr/local/lib/python3.6/site-packages/cudf/dataframe/dataframe.py in query(self, expr, local_dict)
       1905         }
       1906         # Run query
    -> 1907         boolmask = queryutils.query_execute(self, expr, callenv)
       1908 
       1909         selected = Series(boolmask)


    /usr/local/lib/python3.6/site-packages/cudf/utils/queryutils.py in query_execute(df, expr, callenv)
        215             envargs.append(val)
        216     # prepare col args
    --> 217     colarrays = [df[col].to_gpu_array() for col in compiled['colnames']]
        218     # allocate output buffer
        219     nrows = len(df)


    /usr/local/lib/python3.6/site-packages/cudf/utils/queryutils.py in <listcomp>(.0)
        215             envargs.append(val)
        216     # prepare col args
    --> 217     colarrays = [df[col].to_gpu_array() for col in compiled['colnames']]
        218     # allocate output buffer
        219     nrows = len(df)


    /usr/local/lib/python3.6/site-packages/cudf/dataframe/dataframe.py in __getitem__(self, arg)
        230             return self.columns._get_column_major(self, arg)
        231         if isinstance(arg, (str, numbers.Number)) or isinstance(arg, tuple):
    --> 232             s = self._cols[arg]
        233             s.name = arg
        234             s.index = self.index


    KeyError: 'a'


### Method 2:  Simple Columns

#### Filtering Strings by Column Values



```python
# Filtering based on the string column
print(gdf[gdf.strings == 'b'])
```

       integers  strings
    1         2        b


#### Filtering Integers/Floats by Column Values


```python
# Filtering based on the string column
print(gdf[gdf.integers == 2])
```

       integers  strings
    1         2        b


### Method 2:  Simple Rows

#### Filtering by Row Numbers


```python
# Filter rows 0 to 2 (not inclusive of the third row with the index 2)
print(gdf[0:2])
```

       integers  strings
    0         1        a
    1         2        b


### Method 3:  loc[rows, columns]


```python
# The syntax is as follows loc[rows, columns] allowing you to choose rows and columns accordingly
# The example allows us to filter the first 3 rows (inclusive) of the column integers
print(gdf.loc[0:2, ['integers']])
```

       integers
    0         1
    1         2
    2         3



```python

```
