
# RAPIDS cuDF

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ritchieng/deep-learning-wizard/blob/master/docs/machine_learning/gpu/rapids_cudf.ipynb)

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


#### Check GPU if You've Right Version (T4)
Many thanks to NVIDIA team for this snippet of code to automatically set up everything.


```python
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
device_name = pynvml.nvmlDeviceGetName(handle)

if device_name != b'Tesla T4':
  raise Exception("""
    Unfortunately this instance does not have a T4 GPU.
    
    Please make sure you've configured Colab to request a GPU instance type.
    
    Sometimes Colab allocates a Tesla K80 instead of a T4. Resetting the instance.

    If you get a K80 GPU, try Runtime -> Reset all runtimes...
  """)
else:
  print('Woo! You got the right kind of GPU!')
```

    Woo! You got the right kind of GPU!


### Installation of cuDF/cuML
Many thanks to NVIDIA team for this snippet of code to automatically set up everything.


```python
# intall miniconda
!wget -c https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh
!chmod +x Miniconda3-4.5.4-Linux-x86_64.sh
!bash ./Miniconda3-4.5.4-Linux-x86_64.sh -b -f -p /usr/local

# install RAPIDS packages
!conda install -q -y --prefix /usr/local -c conda-forge \
  -c rapidsai-nightly/label/cuda10.0 -c nvidia/label/cuda10.0 \
  cudf cuml

# set environment vars
import sys, os, shutil
sys.path.append('/usr/local/lib/python3.6/site-packages/')
os.environ['NUMBAPRO_NVVM'] = '/usr/local/cuda/nvvm/lib64/libnvvm.so'
os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/local/cuda/nvvm/libdevice/'

# copy .so files to current working dir
for fn in ['libcudf.so', 'librmm.so']:
  shutil.copy('/usr/local/lib/'+fn, os.getcwd())
```

    --2019-05-13 09:31:42--  https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh
    Resolving repo.continuum.io (repo.continuum.io)... 104.18.201.79, 104.18.200.79, 2606:4700::6812:c84f, ...
    Connecting to repo.continuum.io (repo.continuum.io)|104.18.201.79|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 58468498 (56M) [application/x-sh]
    Saving to: ‘Miniconda3-4.5.4-Linux-x86_64.sh’
    
    Miniconda3-4.5.4-Li 100%[===================>]  55.76M  64.9MB/s    in 0.9s    
    
    2019-05-13 09:31:48 (64.9 MB/s) - ‘Miniconda3-4.5.4-Linux-x86_64.sh’ saved [58468498/58468498]
    
    PREFIX=/usr/local
    installing: python-3.6.5-hc3d631a_2 ...
    Python 3.6.5 :: Anaconda, Inc.
    installing: ca-certificates-2018.03.07-0 ...
    installing: conda-env-2.6.0-h36134e3_1 ...
    installing: libgcc-ng-7.2.0-hdf63c60_3 ...
    installing: libstdcxx-ng-7.2.0-hdf63c60_3 ...
    installing: libffi-3.2.1-hd88cf55_4 ...
    installing: ncurses-6.1-hf484d3e_0 ...
    installing: openssl-1.0.2o-h20670df_0 ...
    installing: tk-8.6.7-hc745277_3 ...
    installing: xz-5.2.4-h14c3975_4 ...
    installing: yaml-0.1.7-had09818_2 ...
    installing: zlib-1.2.11-ha838bed_2 ...
    installing: libedit-3.1.20170329-h6b74fdf_2 ...
    installing: readline-7.0-ha6073c6_4 ...
    installing: sqlite-3.23.1-he433501_0 ...
    installing: asn1crypto-0.24.0-py36_0 ...
    installing: certifi-2018.4.16-py36_0 ...
    installing: chardet-3.0.4-py36h0f667ec_1 ...
    installing: idna-2.6-py36h82fb2a8_1 ...
    installing: pycosat-0.6.3-py36h0a5515d_0 ...
    installing: pycparser-2.18-py36hf9f622e_1 ...
    installing: pysocks-1.6.8-py36_0 ...
    installing: ruamel_yaml-0.15.37-py36h14c3975_2 ...
    installing: six-1.11.0-py36h372c433_1 ...
    installing: cffi-1.11.5-py36h9745a5d_0 ...
    installing: setuptools-39.2.0-py36_0 ...
    installing: cryptography-2.2.2-py36h14c3975_0 ...
    installing: wheel-0.31.1-py36_0 ...
    installing: pip-10.0.1-py36_0 ...
    installing: pyopenssl-18.0.0-py36_0 ...
    installing: urllib3-1.22-py36hbe7ace6_0 ...
    installing: requests-2.18.4-py36he2e5f8d_1 ...
    installing: conda-4.5.4-py36_0 ...
    installation finished.
    WARNING:
        You currently have a PYTHONPATH environment variable set. This may cause
        unexpected behavior when running the Python interpreter in Miniconda3.
        For best results, please verify that your PYTHONPATH only points to
        directories of packages that are compatible with the Python interpreter
        in Miniconda3: /usr/local
    Solving environment: ...working... done
    
    ## Package Plan ##
    
      environment location: /usr/local
    
      added / updated specs: 
        - cudf
        - cuml
    
    
    The following packages will be downloaded:
    
        package                    |            build
        ---------------------------|-----------------
        icu-58.2                   |    hf484d3e_1000        22.6 MB  conda-forge
        libgcc-ng-8.2.0            |       hdf63c60_1         7.6 MB
        sqlite-3.26.0              |    h67949de_1001         1.9 MB  conda-forge
        python-3.6.7               |    h381d211_1004        34.5 MB  conda-forge
        numpy-1.16.3               |   py36he5ce36f_0         4.3 MB  conda-forge
        cudatoolkit-10.0.130       |                0       380.0 MB
        libcudf-0.7.0.dev0         |    cuda10.0_1567        21.0 MB  rapidsai-nightly/label/cuda10.0
        parquet-cpp-1.5.1          |                4           3 KB  conda-forge
        arrow-cpp-0.12.1           |   py36h0e61e49_0         6.9 MB  conda-forge
        liblapack-3.8.0            |       9_openblas           6 KB  conda-forge
        libstdcxx-ng-8.2.0         |       hdf63c60_1         2.9 MB
        libblas-3.8.0              |       9_openblas           6 KB  conda-forge
        openssl-1.1.1b             |       h14c3975_1         4.0 MB  conda-forge
        cython-0.29.7              |   py36he1b5a44_0         2.2 MB  conda-forge
        cryptography-2.6.1         |   py36h72c5cf5_0         606 KB  conda-forge
        certifi-2019.3.9           |           py36_0         149 KB  conda-forge
        ca-certificates-2019.3.9   |       hecc5488_0         146 KB  conda-forge
        tk-8.6.9                   |    h84994c4_1001         3.2 MB  conda-forge
        libcuml-0.8.0a             |     cuda10.0_610        24.4 MB  rapidsai-nightly/label/cuda10.0
        bzip2-1.0.6                |    h14c3975_1002         415 KB  conda-forge
        cuml-0.8.0a                |cuda10.0_py36_610         3.0 MB  rapidsai-nightly/label/cuda10.0
        python-dateutil-2.8.0      |             py_0         219 KB  conda-forge
        conda-4.6.14               |           py36_0         2.1 MB  conda-forge
        openblas-0.3.6             |       h6e990d7_2        15.8 MB  conda-forge
        thrift-cpp-0.12.0          |    h0a07b25_1002         2.4 MB  conda-forge
        rmm-0.7.0.dev0             |          py36_54          14 KB  rapidsai-nightly/label/cuda10.0
        pyarrow-0.12.1             |   py36hbbcf98d_0         2.2 MB  conda-forge
        libprotobuf-3.6.1          |    hdbcaa40_1001         4.0 MB  conda-forge
        nvstrings-0.7.0.dev0       |         py36_143          88 KB  rapidsai-nightly/label/cuda10.0
        librmm-0.7.0.dev0          |      cuda10.0_54          39 KB  rapidsai-nightly/label/cuda10.0
        libcblas-3.8.0             |       9_openblas           6 KB  conda-forge
        libgfortran-ng-7.3.0       |       hdf63c60_0         1.3 MB
        libnvstrings-0.7.0.dev0    |     cuda10.0_132         9.3 MB  rapidsai-nightly/label/cuda10.0
        llvmlite-0.28.0            |   py36hdbcaa40_0        20.2 MB  conda-forge
        cudf-0.7.0.dev0            |        py36_1568         2.7 MB  rapidsai-nightly/label/cuda10.0
        pytz-2019.1                |             py_0         227 KB  conda-forge
        numba-0.43.1               |   py36hf2d7682_0         2.9 MB  conda-forge
        boost-cpp-1.68.0           |    h11c811c_1000        20.5 MB  conda-forge
        pandas-0.24.2              |   py36hf484d3e_0        11.1 MB  conda-forge
        libcumlmg-0.0.0.dev0       |     cuda10.0_373         955 KB  nvidia/label/cuda10.0
        ------------------------------------------------------------
                                               Total:       615.7 MB
    
    The following NEW packages will be INSTALLED:
    
        arrow-cpp:       0.12.1-py36h0e61e49_0    conda-forge                    
        boost-cpp:       1.68.0-h11c811c_1000     conda-forge                    
        bzip2:           1.0.6-h14c3975_1002      conda-forge                    
        cudatoolkit:     10.0.130-0                                              
        cudf:            0.7.0.dev0-py36_1568     rapidsai-nightly/label/cuda10.0
        cuml:            0.8.0a-cuda10.0_py36_610 rapidsai-nightly/label/cuda10.0
        cython:          0.29.7-py36he1b5a44_0    conda-forge                    
        icu:             58.2-hf484d3e_1000       conda-forge                    
        libblas:         3.8.0-9_openblas         conda-forge                    
        libcblas:        3.8.0-9_openblas         conda-forge                    
        libcudf:         0.7.0.dev0-cuda10.0_1567 rapidsai-nightly/label/cuda10.0
        libcuml:         0.8.0a-cuda10.0_610      rapidsai-nightly/label/cuda10.0
        libcumlmg:       0.0.0.dev0-cuda10.0_373  nvidia/label/cuda10.0          
        libgfortran-ng:  7.3.0-hdf63c60_0                                        
        liblapack:       3.8.0-9_openblas         conda-forge                    
        libnvstrings:    0.7.0.dev0-cuda10.0_132  rapidsai-nightly/label/cuda10.0
        libprotobuf:     3.6.1-hdbcaa40_1001      conda-forge                    
        librmm:          0.7.0.dev0-cuda10.0_54   rapidsai-nightly/label/cuda10.0
        llvmlite:        0.28.0-py36hdbcaa40_0    conda-forge                    
        numba:           0.43.1-py36hf2d7682_0    conda-forge                    
        numpy:           1.16.3-py36he5ce36f_0    conda-forge                    
        nvstrings:       0.7.0.dev0-py36_143      rapidsai-nightly/label/cuda10.0
        openblas:        0.3.6-h6e990d7_2         conda-forge                    
        pandas:          0.24.2-py36hf484d3e_0    conda-forge                    
        parquet-cpp:     1.5.1-4                  conda-forge                    
        pyarrow:         0.12.1-py36hbbcf98d_0    conda-forge                    
        python-dateutil: 2.8.0-py_0               conda-forge                    
        pytz:            2019.1-py_0              conda-forge                    
        rmm:             0.7.0.dev0-py36_54       rapidsai-nightly/label/cuda10.0
        thrift-cpp:      0.12.0-h0a07b25_1002     conda-forge                    
    
    The following packages will be UPDATED:
    
        ca-certificates: 2018.03.07-0                                             --> 2019.3.9-hecc5488_0  conda-forge
        certifi:         2018.4.16-py36_0                                         --> 2019.3.9-py36_0      conda-forge
        conda:           4.5.4-py36_0                                             --> 4.6.14-py36_0        conda-forge
        cryptography:    2.2.2-py36h14c3975_0                                     --> 2.6.1-py36h72c5cf5_0 conda-forge
        libgcc-ng:       7.2.0-hdf63c60_3                                         --> 8.2.0-hdf63c60_1                
        libstdcxx-ng:    7.2.0-hdf63c60_3                                         --> 8.2.0-hdf63c60_1                
        openssl:         1.0.2o-h20670df_0                                        --> 1.1.1b-h14c3975_1    conda-forge
        python:          3.6.5-hc3d631a_2                                         --> 3.6.7-h381d211_1004  conda-forge
        sqlite:          3.23.1-he433501_0                                        --> 3.26.0-h67949de_1001 conda-forge
        tk:              8.6.7-hc745277_3                                         --> 8.6.9-h84994c4_1001  conda-forge
    
    Preparing transaction: ...working... done
    Verifying transaction: ...working... done
    Executing transaction: ...working... done


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


####Filtering Integers/Floats by Column Values


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

