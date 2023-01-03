# RAPIDS cuDF

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ritchieng/deep-learning-wizard/blob/master/docs/machine_learning/gpu/rapids_cudf.ipynb)

## Environment Setup

### Check Version

#### Python Version


```python
# Check Python Version
!python --version
```

    Python 3.8.16


#### Ubuntu Version


```python
# Check Ubuntu Version
!lsb_release -a
```

    No LSB modules are available.
    Distributor ID:	Ubuntu
    Description:	Ubuntu 18.04.6 LTS
    Release:	18.04
    Codename:	bionic


#### Check CUDA Version


```python
# Check CUDA/cuDNN Version
!nvcc -V && which nvcc
```

    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2021 NVIDIA Corporation
    Built on Sun_Feb_14_21:12:58_PST_2021
    Cuda compilation tools, release 11.2, V11.2.152
    Build cuda_11.2.r11.2/compiler.29618528_0
    /usr/local/cuda/bin/nvcc


#### Check GPU Version


```python
# Check GPU
!nvidia-smi
```

    Tue Jan  3 04:41:06 2023       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
    | N/A   59C    P0    27W /  70W |      0MiB / 15109MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+


#### Setup:
Set up script installs
- Updates gcc in Colab
- Installs Conda
- Install RAPIDS' current stable version of its libraries, as well as some external libraries including:
    - cuDF
    - cuML
    - cuGraph
    - cuSpatial
    - cuSignal
    - BlazingSQL
    - xgboost
- Copy RAPIDS .so files into current working directory, a neccessary workaround for RAPIDS+Colab integration.



```python
# This get the RAPIDS-Colab install files and test check your GPU.  Run this and the next cell only.
# Please read the output of this cell.  If your Colab Instance is not RAPIDS compatible, it will warn you and give you remediation steps.
!git clone https://github.com/rapidsai/rapidsai-csp-utils.git
!python rapidsai-csp-utils/colab/env-check.py
```

    Cloning into 'rapidsai-csp-utils'...
    remote: Enumerating objects: 328, done.[K
    remote: Counting objects: 100% (157/157), done.[K
    remote: Compressing objects: 100% (102/102), done.[K
    remote: Total 328 (delta 92), reused 98 (delta 55), pack-reused 171[K
    Receiving objects: 100% (328/328), 94.64 KiB | 3.05 MiB/s, done.
    Resolving deltas: 100% (154/154), done.
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting pynvml
      Downloading pynvml-11.4.1-py3-none-any.whl (46 kB)
    Installing collected packages: pynvml
    Successfully installed pynvml-11.4.1
    ***********************************************************************
    Woo! Your instance has the right kind of GPU, a Tesla T4!
    We will now install RAPIDS via pip!  Please stand by, should be quick...
    ***********************************************************************
    



```python
# This will update the Colab environment and restart the kernel.  Don't run the next cell until you see the session crash.
!bash rapidsai-csp-utils/colab/update_gcc.sh
import os
os._exit(00)
```

    Updating your Colab environment.  This will restart your kernel.  Don't Panic!
    Get:1 https://cloud.r-project.org/bin/linux/ubuntu bionic-cran40/ InRelease [3,626 B]
    Get:2 http://ppa.launchpad.net/c2d4u.team/c2d4u4.0+/ubuntu bionic InRelease [15.9 kB]
    Get:3 http://security.ubuntu.com/ubuntu bionic-security InRelease [88.7 kB]
    Hit:4 http://archive.ubuntu.com/ubuntu bionic InRelease
    Get:5 http://archive.ubuntu.com/ubuntu bionic-updates InRelease [88.7 kB]
    Hit:6 http://ppa.launchpad.net/cran/libgit2/ubuntu bionic InRelease
    Ign:7 https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64  InRelease
    Hit:8 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  InRelease
    Hit:9 https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64  Release
    Hit:11 http://ppa.launchpad.net/deadsnakes/ppa/ubuntu bionic InRelease
    Hit:12 http://ppa.launchpad.net/graphics-drivers/ppa/ubuntu bionic InRelease
    Get:13 http://archive.ubuntu.com/ubuntu bionic-backports InRelease [83.3 kB]
    Get:14 http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu bionic InRelease [20.8 kB]
    Get:15 http://ppa.launchpad.net/c2d4u.team/c2d4u4.0+/ubuntu bionic/main Sources [2,237 kB]
    Get:16 http://ppa.launchpad.net/c2d4u.team/c2d4u4.0+/ubuntu bionic/main amd64 Packages [1,144 kB]
    Get:17 http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu bionic/main amd64 Packages [50.4 kB]
    Fetched 3,733 kB in 6s (635 kB/s)
    Reading package lists... Done
    Added repo
    Hit:1 https://cloud.r-project.org/bin/linux/ubuntu bionic-cran40/ InRelease
    Hit:2 http://security.ubuntu.com/ubuntu bionic-security InRelease
    Hit:3 http://ppa.launchpad.net/c2d4u.team/c2d4u4.0+/ubuntu bionic InRelease
    Ign:4 https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64  InRelease
    Hit:5 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  InRelease
    Hit:6 https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64  Release
    Hit:7 http://archive.ubuntu.com/ubuntu bionic InRelease
    Hit:9 http://archive.ubuntu.com/ubuntu bionic-updates InRelease
    Hit:10 http://ppa.launchpad.net/cran/libgit2/ubuntu bionic InRelease
    Hit:11 http://archive.ubuntu.com/ubuntu bionic-backports InRelease
    Hit:12 http://ppa.launchpad.net/deadsnakes/ppa/ubuntu bionic InRelease
    Hit:13 http://ppa.launchpad.net/graphics-drivers/ppa/ubuntu bionic InRelease
    Hit:14 http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu bionic InRelease
    Reading package lists... Done
    Installing libstdc++
    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    Selected version '11.1.0-1ubuntu1~18.04.1' (Toolchain test builds:18.04/bionic [amd64]) for 'libstdc++6'
    The following package was automatically installed and is no longer required:
      libnvidia-common-460
    Use 'sudo apt autoremove' to remove it.
    The following additional packages will be installed:
      gcc-11-base libgcc-s1
    The following NEW packages will be installed:
      gcc-11-base libgcc-s1
    The following packages will be upgraded:
      libstdc++6
    1 upgraded, 2 newly installed, 0 to remove and 31 not upgraded.
    Need to get 641 kB of archives.
    After this operation, 981 kB of additional disk space will be used.
    Get:1 http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu bionic/main amd64 gcc-11-base amd64 11.1.0-1ubuntu1~18.04.1 [19.0 kB]
    Get:2 http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu bionic/main amd64 libgcc-s1 amd64 11.1.0-1ubuntu1~18.04.1 [41.8 kB]
    Get:3 http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu bionic/main amd64 libstdc++6 amd64 11.1.0-1ubuntu1~18.04.1 [580 kB]
    Fetched 641 kB in 2s (277 kB/s)
    debconf: unable to initialize frontend: Dialog
    debconf: (No usable dialog-like program is installed, so the dialog based frontend cannot be used. at /usr/share/perl5/Debconf/FrontEnd/Dialog.pm line 76, <> line 3.)
    debconf: falling back to frontend: Readline
    debconf: unable to initialize frontend: Readline
    debconf: (This frontend requires a controlling tty.)
    debconf: falling back to frontend: Teletype
    dpkg-preconfigure: unable to re-open stdin: 
    Selecting previously unselected package gcc-11-base:amd64.
    (Reading database ... 124016 files and directories currently installed.)
    Preparing to unpack .../gcc-11-base_11.1.0-1ubuntu1~18.04.1_amd64.deb ...
    Unpacking gcc-11-base:amd64 (11.1.0-1ubuntu1~18.04.1) ...
    Setting up gcc-11-base:amd64 (11.1.0-1ubuntu1~18.04.1) ...
    Selecting previously unselected package libgcc-s1:amd64.
    (Reading database ... 124021 files and directories currently installed.)
    Preparing to unpack .../libgcc-s1_11.1.0-1ubuntu1~18.04.1_amd64.deb ...
    Unpacking libgcc-s1:amd64 (11.1.0-1ubuntu1~18.04.1) ...
    Replacing files in old package libgcc1:amd64 (1:8.4.0-1ubuntu1~18.04) ...
    Setting up libgcc-s1:amd64 (11.1.0-1ubuntu1~18.04.1) ...
    (Reading database ... 124023 files and directories currently installed.)
    Preparing to unpack .../libstdc++6_11.1.0-1ubuntu1~18.04.1_amd64.deb ...
    Unpacking libstdc++6:amd64 (11.1.0-1ubuntu1~18.04.1) over (8.4.0-1ubuntu1~18.04) ...
    Setting up libstdc++6:amd64 (11.1.0-1ubuntu1~18.04.1) ...
    Processing triggers for libc-bin (2.27-3ubuntu1.6) ...
    restarting Colab...



```python
# This will install CondaColab.  This will restart your kernel one last time.  Run this cell by itself and only run the next cell once you see the session crash.
import condacolab
condacolab.install()
```

    ⏬ Downloading https://github.com/jaimergp/miniforge/releases/latest/download/Mambaforge-colab-Linux-x86_64.sh...
    📦 Installing...
    📌 Adjusting configuration...
    🩹 Patching environment...
    ⏲ Done in 0:00:14
    🔁 Restarting kernel...



```python
# you can now run the rest of the cells as normal
import condacolab
condacolab.check()
```

    ✨🍰✨ Everything looks OK!


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
#import nvstrings, nvcategory, cudf
import cudf
import cuml
import os
import numpy as np
import pandas as pd
```

    /usr/local/lib/python3.8/site-packages/cupy/_environment.py:439: UserWarning: 
    --------------------------------------------------------------------------------
    
      CuPy may not function correctly because multiple CuPy packages are installed
      in your environment:
    
        cupy, cupy-cuda11x
    
      Follow these steps to resolve this issue:
    
        1. For all packages listed above, run the following command to remove all
           existing CuPy installations:
    
             $ pip uninstall <package_name>
    
          If you previously installed CuPy via conda, also run the following:
    
             $ conda uninstall cupy
    
        2. Install the appropriate CuPy package.
           Refer to the Installation Guide for detailed instructions.
    
             https://docs.cupy.dev/en/stable/install.html
    
    --------------------------------------------------------------------------------
    
      warnings.warn(f'''


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
    <class 'cudf.core.series.Series'>


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

# Using a dictionary of key-value pairs
# Each key in the dictionary represents a category
# The key is the category's name
# The value is a list of the values in that category
gdf = cudf.DataFrame({
    # Create 10 busindates ess from 1st January 2019 via pandas
    'dates': pd.date_range('1/1/2019', periods=10, freq='B'),
    # Integers
    'integers': [i for i in range(10)],
    # Floats
    'floats': [float(i) for i in range(10)]
})

# Print dataframe
print(gdf)
```

           dates  integers  floats
    0 2019-01-01         0     0.0
    1 2019-01-02         1     1.0
    2 2019-01-03         2     2.0
    3 2019-01-04         3     3.0
    4 2019-01-07         4     4.0
    5 2019-01-08         5     5.0
    6 2019-01-09         6     6.0
    7 2019-01-10         7     7.0
    8 2019-01-11         8     8.0
    9 2019-01-14         9     9.0


### Create 2 column Dataframe
- Consisting of integers and string category


```python
# Using a dictionary
# Each key in the dictionary represents a category
# The key is the category's name
# The value is a list of the values in that category
gdf = cudf.DataFrame({
    'integers': [1 ,2, 3, 4],
    'string': ['a', 'b', 'c', 'd']
})

print(gdf)
```

       integers string
    0         1      a
    1         2      b
    2         3      c
    3         4      d


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

       integers strings
    0         1       a
    1         2       b
    2         3       c
    3         4       d


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

       integers strings
    0         1       a
    1         2       b


### Viewing Bottom of DataFrame


```python
num_of_rows_to_view = 3 
print(gdf.tail(num_of_rows_to_view))
```

       integers strings
    1         2       b
    2         3       c
    3         4       d


## Filtering

### Method 1: Query

#### Filtering Integers/Floats by Column Values
- This only works for floats and integers, not for strings


```python
# DO NOT RUN
# TOFIX: `cffi` package version mismatch error
print(gdf.query('integers == 1'))
```

#### Filtering Strings by Column Values
- This only works for floats and integers, not for strings so this will return an error!


```python
print(gdf.query('strings == a'))
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    /usr/local/lib/python3.8/site-packages/cudf/core/dataframe.py in extract_col(df, col)
       7558     try:
    -> 7559         return df._data[col]
       7560     except KeyError:


    /usr/local/lib/python3.8/site-packages/cudf/core/column_accessor.py in __getitem__(self, key)
        154     def __getitem__(self, key: Any) -> ColumnBase:
    --> 155         return self._data[key]
        156 


    KeyError: 'a'

    
    During handling of the above exception, another exception occurred:


    KeyError                                  Traceback (most recent call last)

    <ipython-input-22-5cfd0345d51c> in <module>
    ----> 1 print(gdf.query('strings == a'))
    

    /usr/local/lib/python3.8/site-packages/cudf/core/dataframe.py in query(self, expr, local_dict)
       4172             }
       4173             # Run query
    -> 4174             boolmask = queryutils.query_execute(self, expr, callenv)
       4175             return self._apply_boolean_mask(boolmask)
       4176 


    /usr/local/lib/python3.8/site-packages/cudf/utils/queryutils.py in query_execute(df, expr, callenv)
        212 
        213     # prepare col args
    --> 214     colarrays = [cudf.core.dataframe.extract_col(df, col) for col in columns]
        215 
        216     # wait to check the types until we know which cols are used


    /usr/local/lib/python3.8/site-packages/cudf/utils/queryutils.py in <listcomp>(.0)
        212 
        213     # prepare col args
    --> 214     colarrays = [cudf.core.dataframe.extract_col(df, col) for col in columns]
        215 
        216     # wait to check the types until we know which cols are used


    /usr/local/lib/python3.8/site-packages/cudf/core/dataframe.py in extract_col(df, col)
       7565         ):
       7566             return df.index._data.columns[0]
    -> 7567         return df.index._data[col]
       7568 
       7569 


    /usr/local/lib/python3.8/site-packages/cudf/core/column_accessor.py in __getitem__(self, key)
        153 
        154     def __getitem__(self, key: Any) -> ColumnBase:
    --> 155         return self._data[key]
        156 
        157     def __setitem__(self, key: Any, value: Any):


    KeyError: 'a'


### Method 2:  Simple Columns

#### Filtering Strings by Column Values



```python
# DO NOT RUN
# TOFIX: `cffi` package version mismatch error
# Filtering based on the string column
print(gdf[gdf.strings == 'b'])
```

#### Filtering Integers/Floats by Column Values


```python
# Filtering based on the string column
print(gdf[gdf.integers == 2])
```

       integers strings
    1         2       b


### Method 2:  Simple Rows

#### Filtering by Row Numbers


```python
# Filter rows 0 to 2 (not inclusive of the third row with the index 2)
print(gdf[0:2])
```

       integers strings
    0         1       a
    1         2       b


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
