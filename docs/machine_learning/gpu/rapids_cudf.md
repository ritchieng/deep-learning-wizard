
# RAPIDS cuDF

!!! tip "Run Jupyter Notebook"
    You can run the code for this section in this [jupyter notebook link](https://github.com/ritchieng/deep-learning-wizard/blob/master/docs/machine_learning/gpu/rapids_cudf.ipynb) on Google Colab. Simply copy the notebook into your Google Drive and run with Google Colab.

## Environment Setup

### Check Version

#### Python Version


```python
# Check Python Version
!python --version
```

    Python 3.6.7


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

    Wed Apr 24 07:41:30 2019       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 418.56       Driver Version: 410.79       CUDA Version: 10.0     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
    | N/A   75C    P8    18W /  70W |      0MiB / 15079MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+


### Installation of cuDF/cuML


```python
!python -m pip install cudf-cuda100==0.6.1 cuml-cuda100==0.6.1
```

    Requirement already satisfied: cudf-cuda100==0.6.1 in /usr/local/lib/python3.6/dist-packages (0.6.1)
    Requirement already satisfied: cuml-cuda100==0.6.1 in /usr/local/lib/python3.6/dist-packages (0.6.1)
    Requirement already satisfied: numpy>=1.14 in /usr/local/lib/python3.6/dist-packages (from cudf-cuda100==0.6.1) (1.16.2)
    Requirement already satisfied: numba<0.42,>=0.40.0 in /usr/local/lib/python3.6/dist-packages (from cudf-cuda100==0.6.1) (0.40.1)
    Requirement already satisfied: pyarrow==0.12.1 in /usr/local/lib/python3.6/dist-packages (from cudf-cuda100==0.6.1) (0.12.1)
    Requirement already satisfied: cffi>=1.0.0 in /usr/local/lib/python3.6/dist-packages (from cudf-cuda100==0.6.1) (1.12.3)
    Requirement already satisfied: pandas>=0.23.4 in /usr/local/lib/python3.6/dist-packages (from cudf-cuda100==0.6.1) (0.24.2)
    Requirement already satisfied: pycparser==2.19 in /usr/local/lib/python3.6/dist-packages (from cudf-cuda100==0.6.1) (2.19)
    Requirement already satisfied: nvstrings-cuda100 in /usr/local/lib/python3.6/dist-packages (from cudf-cuda100==0.6.1) (0.3.0.post1)
    Requirement already satisfied: cython<0.30,>=0.29 in /usr/local/lib/python3.6/dist-packages (from cudf-cuda100==0.6.1) (0.29.7)
    Requirement already satisfied: llvmlite>=0.25.0dev0 in /usr/local/lib/python3.6/dist-packages (from numba<0.42,>=0.40.0->cudf-cuda100==0.6.1) (0.28.0)
    Requirement already satisfied: six>=1.0.0 in /usr/local/lib/python3.6/dist-packages (from pyarrow==0.12.1->cudf-cuda100==0.6.1) (1.12.0)
    Requirement already satisfied: pytz>=2011k in /usr/local/lib/python3.6/dist-packages (from pandas>=0.23.4->cudf-cuda100==0.6.1) (2018.9)
    Requirement already satisfied: python-dateutil>=2.5.0 in /usr/local/lib/python3.6/dist-packages (from pandas>=0.23.4->cudf-cuda100==0.6.1) (2.5.3)


### Installation of NVIDIA Toolkit and Numba


```python
# Install CUDA toolkit
!apt install -y --no-install-recommends -q nvidia-cuda-toolkit
!pip install numba

import os
os.environ['NUMBAPRO_LIBDEVICE'] = "/usr/lib/nvidia-cuda-toolkit/libdevice"
os.environ['NUMBAPRO_NVVM'] = "/usr/lib/x86_64-linux-gnu/libnvvm.so"
```

    Reading package lists...
    Building dependency tree...
    Reading state information...
    nvidia-cuda-toolkit is already the newest version (9.1.85-3ubuntu1).
    The following package was automatically installed and is no longer required:
      libnvidia-common-410
    Use 'apt autoremove' to remove it.
    0 upgraded, 0 newly installed, 0 to remove and 6 not upgraded.
    Requirement already satisfied: numba in /usr/local/lib/python3.6/dist-packages (0.40.1)
    Requirement already satisfied: llvmlite>=0.25.0dev0 in /usr/local/lib/python3.6/dist-packages (from numba) (0.28.0)
    Requirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from numba) (1.16.2)


## Critical Imports


```python
# Critical imports
import os
import numpy as np
import pandas as pd
import cudf
```

## DataFrame Operations


```python
df = cudf.Series([1, 2, 3, 4, 5, 6])
print(df)
print(type(df))
```

    0    1
    1    2
    2    3
    3    4
    4    5
    5    6
    dtype: int64
    <class 'cudf.dataframe.series.Series'>


### Create a single column dataframe of floats


```python
df = cudf.Series([1., 2., 3., 4., 5., 6.])
print(df)
```

    0    1.0
    1    2.0
    2    3.0
    3    4.0
    4    5.0
    5    6.0
    dtype: float64


### Create three column dataframe of dates, integers and floats


```python
# Import
import datetime as dt

# Create blank cudf dataframe
df = cudf.DataFrame()

# Create 10 busindates ess from 1st January 2019 via pandas
df['dates'] = pd.date_range('1/1/2019', periods=10, freq='B')

# Integers
df['integers'] = [i for i in range(10)]

# Floats
df['floats'] = [float(i) for i in range(10)]

# Print dataframe
print(df)
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


### Create a dataframe of alphabets a, b and c (strings)


```python
s = cudf.Series(['a', 'b', 'c'])
print(s)
```

    0    a
    1    b
    2    c
    dtype: object


### Create a 2 Column Dataframe of integers and string category
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
df = cudf.DataFrame.from_pandas(pandas_df)

# Print dataframe
print(df)
```

       integers  strings
    0         1        a
    1         2        b
    2         3        c
    3         4        d


### Printing Column Names


```python
df.columns
```




    Index(['integers', 'strings'], dtype='object')



### Filtering Integers/Floats by Column Values (Method 1
- This only works for floats and integers, not for strings


```python
print(df.query('integers == 1'))
```

       integers  strings
    0         1        a


### Filtering Strings by Column Values (Method 1)


```python
print(df.query('strings == a'))
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-15-8bd9757c1b26> in <module>()
    ----> 1 print(df.query('strings == a'))
    

    /usr/local/lib/python3.6/dist-packages/cudf/dataframe/dataframe.py in query(self, expr)
       1787         }
       1788         # Run query
    -> 1789         boolmask = queryutils.query_execute(self, expr, callenv)
       1790 
       1791         selected = Series(boolmask)


    /usr/local/lib/python3.6/dist-packages/cudf/utils/queryutils.py in query_execute(df, expr, callenv)
        214             envargs.append(val)
        215     # prepare col args
    --> 216     colarrays = [df[col].to_gpu_array() for col in compiled['colnames']]
        217     # allocate output buffer
        218     nrows = len(df)


    /usr/local/lib/python3.6/dist-packages/cudf/utils/queryutils.py in <listcomp>(.0)
        214             envargs.append(val)
        215     # prepare col args
    --> 216     colarrays = [df[col].to_gpu_array() for col in compiled['colnames']]
        217     # allocate output buffer
        218     nrows = len(df)


    /usr/local/lib/python3.6/dist-packages/cudf/dataframe/dataframe.py in __getitem__(self, arg)
        212         if isinstance(arg, str) or isinstance(arg, numbers.Integral) or \
        213            isinstance(arg, tuple):
    --> 214             s = self._cols[arg]
        215             s.name = arg
        216             return s


    KeyError: 'a'


### Filtering Strings by Column Values (Method 2)



```python
# Filtering based on the string column
print(df[df.strings == 'b'])
```

       integers  strings
    1         2        b


### Filtering Integers/Floats by Column Values (Method 2)


```python
# Filtering based on the string column
print(df[df.integers == 2])
```

       integers  strings
    1         2        b

