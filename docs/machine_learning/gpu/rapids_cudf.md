---
comments: true
---

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

    Wed Jan  4 19:14:22 2023       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
    | N/A   48C    P0    29W /  70W |      0MiB / 15109MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+


#Setup:
This set up script:

1. Checks to make sure that the GPU is RAPIDS compatible
1. Installs the **current stable version** of RAPIDSAI's core libraries using pip, which are:
  1. cuDF
  1. cuML
  1. cuGraph
  1. xgboost

**This will complete in about 3-4 minutes**

Please use the [RAPIDS Conda Colab Template notebook](https://colab.research.google.com/drive/1TAAi_szMfWqRfHVfjGSqnGVLr_ztzUM9) if you need to install any of RAPIDS Extended libraries, such as:
- cuSpatial
- cuSignal
- cuxFilter
- cuCIM

OR
- nightly versions of any library 


```python
# This get the RAPIDS-Colab install files and test check your GPU.  Run this and the next cell only.
# Please read the output of this cell.  If your Colab Instance is not RAPIDS compatible, it will warn you and give you remediation steps.
!git clone https://github.com/rapidsai/rapidsai-csp-utils.git
!python rapidsai-csp-utils/colab/pip-install.py
```

    Cloning into 'rapidsai-csp-utils'...
    remote: Enumerating objects: 328, done.[K
    remote: Counting objects: 100% (157/157), done.[K
    remote: Compressing objects: 100% (102/102), done.[K
    remote: Total 328 (delta 92), reused 98 (delta 55), pack-reused 171[K
    Receiving objects: 100% (328/328), 94.64 KiB | 18.93 MiB/s, done.
    Resolving deltas: 100% (154/154), done.
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting pynvml
      Downloading pynvml-11.4.1-py3-none-any.whl (46 kB)
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 47.0/47.0 KB 6.1 MB/s eta 0:00:00
    Installing collected packages: pynvml
    Successfully installed pynvml-11.4.1
    ***********************************************************************
    Woo! Your instance has the right kind of GPU, a Tesla T4!
    We will now install RAPIDS via pip!  Please stand by, should be quick...
    ***********************************************************************
    
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/, https://pypi.ngc.nvidia.com
    Collecting cudf-cu11
      Downloading https://developer.download.nvidia.com/compute/redist/cudf-cu11/cudf_cu11-22.12.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (442.8 MB)
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 442.8/442.8 MB 3.5 MB/s eta 0:00:00
    Collecting cuml-cu11
      Downloading https://developer.download.nvidia.com/compute/redist/cuml-cu11/cuml_cu11-22.12.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1359.8 MB)
    tcmalloc: large alloc 1359798272 bytes == 0x3116000 @  0x7f53812b21e7 0x4d30a0 0x4d312c 0x5d6f4c 0x51edd1 0x51ef5b 0x4f750a 0x4997a2 0x55cd91 0x5d8941 0x4997a2 0x55cd91 0x5d8941 0x4997a2 0x55cd91 0x5d8941 0x4997a2 0x55cd91 0x5d8941 0x4997a2 0x55cd91 0x5d8941 0x4997a2 0x5d8868 0x4997a2 0x55cd91 0x5d8941 0x49abe4 0x55cd91 0x5d8941 0x4997a2
    tcmalloc: large alloc 1699749888 bytes == 0x541e4000 @  0x7f53812b3615 0x5d6f4c 0x51edd1 0x51ef5b 0x4f750a 0x4997a2 0x55cd91 0x5d8941 0x4997a2 0x55cd91 0x5d8941 0x4997a2 0x55cd91 0x5d8941 0x4997a2 0x55cd91 0x5d8941 0x4997a2 0x55cd91 0x5d8941 0x4997a2 0x5d8868 0x4997a2 0x55cd91 0x5d8941 0x49abe4 0x55cd91 0x5d8941 0x4997a2 0x55cd91 0x5d8941
    tcmalloc: large alloc 1359798272 bytes == 0x3116000 @  0x7f53812b21e7 0x4d30a0 0x5dede2 0x6758aa 0x4f750a 0x4997a2 0x55cd91 0x5d8941 0x4997a2 0x55cd91 0x5d8941 0x4fe318 0x5da092 0x62042c 0x5d8d8c 0x561f80 0x4fd2db 0x4997c7 0x4fd8b5 0x4997c7 0x4fd8b5 0x49abe4 0x4f5fe9 0x55e146 0x4f5fe9 0x55e146 0x4f5fe9 0x55e146 0x5d8868 0x5da092 0x587116
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.4/1.4 GB 1.3 MB/s eta 0:00:00
    Collecting cugraph-cu11
      Downloading https://developer.download.nvidia.com/compute/redist/cugraph-cu11/cugraph_cu11-22.12.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1028.4 MB)
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/1.0 GB 1.9 MB/s eta 0:00:00
    Requirement already satisfied: numba>=0.56.2 in /usr/local/lib/python3.8/dist-packages (from cudf-cu11) (0.56.4)
    Requirement already satisfied: numpy in /usr/local/lib/python3.8/dist-packages (from cudf-cu11) (1.21.6)
    Collecting ptxcompiler-cu11
      Downloading https://developer.download.nvidia.com/compute/redist/ptxcompiler-cu11/ptxcompiler_cu11-0.7.0.post1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (8.8 MB)
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.8/8.8 MB 99.1 MB/s eta 0:00:00
    Collecting cuda-python<12.0,>=11.7.1
      Downloading cuda_python-11.8.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (16.2 MB)
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.2/16.2 MB 77.6 MB/s eta 0:00:00
    Requirement already satisfied: pyarrow==9.0.0 in /usr/local/lib/python3.8/dist-packages (from cudf-cu11) (9.0.0)
    Requirement already satisfied: pandas<1.6.0dev0,>=1.0 in /usr/local/lib/python3.8/dist-packages (from cudf-cu11) (1.3.5)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.8/dist-packages (from cudf-cu11) (4.4.0)
    Requirement already satisfied: cupy-cuda11x in /usr/local/lib/python3.8/dist-packages (from cudf-cu11) (11.0.0)
    Requirement already satisfied: cachetools in /usr/local/lib/python3.8/dist-packages (from cudf-cu11) (5.2.0)
    Requirement already satisfied: fsspec>=0.6.0 in /usr/local/lib/python3.8/dist-packages (from cudf-cu11) (2022.11.0)
    Collecting protobuf<3.21.0a0,>=3.20.1
      Downloading protobuf-3.20.3-cp38-cp38-manylinux_2_5_x86_64.manylinux1_x86_64.whl (1.0 MB)
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/1.0 MB 46.4 MB/s eta 0:00:00
    Collecting rmm-cu11
      Downloading https://developer.download.nvidia.com/compute/redist/rmm-cu11/rmm_cu11-22.12.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.8 MB)
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.8/1.8 MB 61.1 MB/s eta 0:00:00
    Collecting cubinlinker-cu11
      Downloading https://developer.download.nvidia.com/compute/redist/cubinlinker-cu11/cubinlinker_cu11-0.3.0.post1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (8.8 MB)
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.8/8.8 MB 99.9 MB/s eta 0:00:00
    Requirement already satisfied: packaging in /usr/local/lib/python3.8/dist-packages (from cudf-cu11) (21.3)
    Collecting nvtx>=0.2.1
      Downloading nvtx-0.2.5-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (453 kB)
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 453.6/453.6 KB 28.4 MB/s eta 0:00:00
    Requirement already satisfied: seaborn in /usr/local/lib/python3.8/dist-packages (from cuml-cu11) (0.11.2)
    Collecting raft-dask-cu11
      Downloading https://developer.download.nvidia.com/compute/redist/raft-dask-cu11/raft_dask_cu11-22.12.0.post1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (210.5 MB)
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 210.5/210.5 MB 6.7 MB/s eta 0:00:00
    Requirement already satisfied: scipy in /usr/local/lib/python3.8/dist-packages (from cuml-cu11) (1.7.3)
    Collecting treelite==3.0.1
      Downloading treelite-3.0.1-py3-none-manylinux2014_x86_64.whl (864 kB)
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 864.6/864.6 KB 38.0 MB/s eta 0:00:00
    Collecting treelite-runtime==3.0.1
      Downloading treelite_runtime-3.0.1-py3-none-manylinux2014_x86_64.whl (191 kB)
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 191.9/191.9 KB 25.3 MB/s eta 0:00:00
    Collecting dask-cudf-cu11
      Downloading https://developer.download.nvidia.com/compute/redist/dask-cudf-cu11/dask_cudf_cu11-22.12.0.post1-py3-none-any.whl (76 kB)
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 76.6/76.6 KB 12.2 MB/s eta 0:00:00
    Collecting pylibraft-cu11
      Downloading https://developer.download.nvidia.com/compute/redist/pylibraft-cu11/pylibraft_cu11-22.12.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (580.3 MB)
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 580.3/580.3 MB 3.2 MB/s eta 0:00:00
    Collecting pylibcugraph-cu11
      Downloading https://developer.download.nvidia.com/compute/redist/pylibcugraph-cu11/pylibcugraph_cu11-22.12.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1627.2 MB)
    tcmalloc: large alloc 1627185152 bytes == 0x541e8000 @  0x7f53812b21e7 0x4d30a0 0x4d312c 0x5d6f4c 0x51edd1 0x51ef5b 0x4f750a 0x4997a2 0x55cd91 0x5d8941 0x4997a2 0x55cd91 0x5d8941 0x4997a2 0x55cd91 0x5d8941 0x4997a2 0x55cd91 0x5d8941 0x4997a2 0x55cd91 0x5d8941 0x4997a2 0x5d8868 0x4997a2 0x55cd91 0x5d8941 0x49abe4 0x55cd91 0x5d8941 0x4997a2
    tcmalloc: large alloc 2033983488 bytes == 0xb51b6000 @  0x7f53812b3615 0x5d6f4c 0x51edd1 0x51ef5b 0x4f750a 0x4997a2 0x55cd91 0x5d8941 0x4997a2 0x55cd91 0x5d8941 0x4997a2 0x55cd91 0x5d8941 0x4997a2 0x55cd91 0x5d8941 0x4997a2 0x55cd91 0x5d8941 0x4997a2 0x5d8868 0x4997a2 0x55cd91 0x5d8941 0x49abe4 0x55cd91 0x5d8941 0x4997a2 0x55cd91 0x5d8941
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.6/1.6 GB 1.1 MB/s eta 0:00:00
    Collecting dask-cuda
      Downloading dask_cuda-22.12.0-py3-none-any.whl (121 kB)
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 121.1/121.1 KB 17.5 MB/s eta 0:00:00
    Requirement already satisfied: cython in /usr/local/lib/python3.8/dist-packages (from cuda-python<12.0,>=11.7.1->cudf-cu11) (0.29.32)
    Requirement already satisfied: importlib-metadata in /usr/local/lib/python3.8/dist-packages (from numba>=0.56.2->cudf-cu11) (5.2.0)
    Requirement already satisfied: llvmlite<0.40,>=0.39.0dev0 in /usr/local/lib/python3.8/dist-packages (from numba>=0.56.2->cudf-cu11) (0.39.1)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.8/dist-packages (from numba>=0.56.2->cudf-cu11) (57.4.0)
    Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.8/dist-packages (from pandas<1.6.0dev0,>=1.0->cudf-cu11) (2.8.2)
    Requirement already satisfied: pytz>=2017.3 in /usr/local/lib/python3.8/dist-packages (from pandas<1.6.0dev0,>=1.0->cudf-cu11) (2022.7)
    Requirement already satisfied: fastrlock>=0.5 in /usr/local/lib/python3.8/dist-packages (from cupy-cuda11x->cudf-cu11) (0.8.1)
    Requirement already satisfied: zict>=0.1.3 in /usr/local/lib/python3.8/dist-packages (from dask-cuda->cugraph-cu11) (2.2.0)
    Collecting distributed==2022.11.1
      Downloading distributed-2022.11.1-py3-none-any.whl (923 kB)
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 923.4/923.4 KB 50.7 MB/s eta 0:00:00
    Collecting dask==2022.11.1
      Downloading dask-2022.11.1-py3-none-any.whl (1.1 MB)
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.1/1.1 MB 52.6 MB/s eta 0:00:00
    Requirement already satisfied: pynvml>=11.0.0 in /usr/local/lib/python3.8/dist-packages (from dask-cuda->cugraph-cu11) (11.4.1)
    Requirement already satisfied: click>=7.0 in /usr/local/lib/python3.8/dist-packages (from dask==2022.11.1->dask-cuda->cugraph-cu11) (7.1.2)
    Requirement already satisfied: toolz>=0.8.2 in /usr/local/lib/python3.8/dist-packages (from dask==2022.11.1->dask-cuda->cugraph-cu11) (0.12.0)
    Requirement already satisfied: partd>=0.3.10 in /usr/local/lib/python3.8/dist-packages (from dask==2022.11.1->dask-cuda->cugraph-cu11) (1.3.0)
    Requirement already satisfied: cloudpickle>=1.1.1 in /usr/local/lib/python3.8/dist-packages (from dask==2022.11.1->dask-cuda->cugraph-cu11) (1.5.0)
    Requirement already satisfied: pyyaml>=5.3.1 in /usr/local/lib/python3.8/dist-packages (from dask==2022.11.1->dask-cuda->cugraph-cu11) (6.0)
    Requirement already satisfied: tornado<6.2,>=6.0.3 in /usr/local/lib/python3.8/dist-packages (from distributed==2022.11.1->dask-cuda->cugraph-cu11) (6.0.4)
    Requirement already satisfied: locket>=1.0.0 in /usr/local/lib/python3.8/dist-packages (from distributed==2022.11.1->dask-cuda->cugraph-cu11) (1.0.0)
    Requirement already satisfied: tblib>=1.6.0 in /usr/local/lib/python3.8/dist-packages (from distributed==2022.11.1->dask-cuda->cugraph-cu11) (1.7.0)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.8/dist-packages (from distributed==2022.11.1->dask-cuda->cugraph-cu11) (2.11.3)
    Requirement already satisfied: psutil>=5.0 in /usr/local/lib/python3.8/dist-packages (from distributed==2022.11.1->dask-cuda->cugraph-cu11) (5.4.8)
    Requirement already satisfied: urllib3 in /usr/local/lib/python3.8/dist-packages (from distributed==2022.11.1->dask-cuda->cugraph-cu11) (1.24.3)
    Requirement already satisfied: sortedcontainers!=2.0.0,!=2.0.1 in /usr/local/lib/python3.8/dist-packages (from distributed==2022.11.1->dask-cuda->cugraph-cu11) (2.4.0)
    Requirement already satisfied: msgpack>=0.6.0 in /usr/local/lib/python3.8/dist-packages (from distributed==2022.11.1->dask-cuda->cugraph-cu11) (1.0.4)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.8/dist-packages (from packaging->cudf-cu11) (3.0.9)
    Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.8/dist-packages (from raft-dask-cu11->cuml-cu11) (1.2.0)
    Collecting ucx-py-cu11
      Downloading https://developer.download.nvidia.com/compute/redist/ucx-py-cu11/ucx_py_cu11-0.29.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (8.3 MB)
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.3/8.3 MB 72.8 MB/s eta 0:00:00
    Requirement already satisfied: matplotlib>=2.2 in /usr/local/lib/python3.8/dist-packages (from seaborn->cuml-cu11) (3.2.2)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.8/dist-packages (from matplotlib>=2.2->seaborn->cuml-cu11) (0.11.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.8/dist-packages (from matplotlib>=2.2->seaborn->cuml-cu11) (1.4.4)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.8/dist-packages (from python-dateutil>=2.7.3->pandas<1.6.0dev0,>=1.0->cudf-cu11) (1.15.0)
    Requirement already satisfied: heapdict in /usr/local/lib/python3.8/dist-packages (from zict>=0.1.3->dask-cuda->cugraph-cu11) (1.0.1)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.8/dist-packages (from importlib-metadata->numba>=0.56.2->cudf-cu11) (3.11.0)
    Requirement already satisfied: MarkupSafe>=0.23 in /usr/local/lib/python3.8/dist-packages (from jinja2->distributed==2022.11.1->dask-cuda->cugraph-cu11) (2.0.1)
    Installing collected packages: ptxcompiler-cu11, nvtx, cubinlinker-cu11, ucx-py-cu11, protobuf, cuda-python, treelite-runtime, treelite, dask, rmm-cu11, distributed, pylibraft-cu11, dask-cuda, cudf-cu11, raft-dask-cu11, pylibcugraph-cu11, dask-cudf-cu11, cuml-cu11, cugraph-cu11
      Attempting uninstall: protobuf
        Found existing installation: protobuf 3.19.6
        Uninstalling protobuf-3.19.6:
          Successfully uninstalled protobuf-3.19.6
      Attempting uninstall: dask
        Found existing installation: dask 2022.2.1
        Uninstalling dask-2022.2.1:
          Successfully uninstalled dask-2022.2.1
      Attempting uninstall: distributed
        Found existing installation: distributed 2022.2.1
        Uninstalling distributed-2022.2.1:
          Successfully uninstalled distributed-2022.2.1
    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    tensorflow 2.9.2 requires protobuf<3.20,>=3.9.2, but you have protobuf 3.20.3 which is incompatible.
    tensorboard 2.9.1 requires protobuf<3.20,>=3.9.2, but you have protobuf 3.20.3 which is incompatible.
    Successfully installed cubinlinker-cu11-0.3.0.post1 cuda-python-11.8.1 cudf-cu11-22.12.0 cugraph-cu11-22.12.0 cuml-cu11-22.12.0 dask-2022.11.1 dask-cuda-22.12.0 dask-cudf-cu11-22.12.0.post1 distributed-2022.11.1 nvtx-0.2.5 protobuf-3.20.3 ptxcompiler-cu11-0.7.0.post1 pylibcugraph-cu11-22.12.0 pylibraft-cu11-22.12.0 raft-dask-cu11-22.12.0.post1 rmm-cu11-22.12.0 treelite-3.0.1 treelite-runtime-3.0.1 ucx-py-cu11-0.29.0
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting cupy-cuda11x
      Downloading cupy_cuda11x-11.4.0-cp38-cp38-manylinux1_x86_64.whl (93.7 MB)
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 93.7/93.7 MB 10.7 MB/s eta 0:00:00
    Requirement already satisfied: numpy<1.26,>=1.20 in /usr/local/lib/python3.8/dist-packages (from cupy-cuda11x) (1.21.6)
    Requirement already satisfied: fastrlock>=0.5 in /usr/local/lib/python3.8/dist-packages (from cupy-cuda11x) (0.8.1)
    Installing collected packages: cupy-cuda11x
    Successfully installed cupy-cuda11x-11.4.0
    
              ***********************************************************************
              With the new pip install complete, please do not run any further installation 
              commands from the conda based installation methods!!!  
              
              In your personal files, you can delete these cells.  
              
              RAPIDSAI owned templates/notebooks should already be updated with no action needed.
              ***********************************************************************
              


## Critical Imports


```python
# Critical imports
import cudf
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

       integers strings
    0         1       a


#### Filtering Strings by Column Values
- This only works for floats and integers, not for strings so this will return an error!


```python
print(gdf.query('strings == a'))
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    /usr/local/lib/python3.8/dist-packages/cudf/core/dataframe.py in extract_col(df, col)
       7558     try:
    -> 7559         return df._data[col]
       7560     except KeyError:


    /usr/local/lib/python3.8/dist-packages/cudf/core/column_accessor.py in __getitem__(self, key)
        154     def __getitem__(self, key: Any) -> ColumnBase:
    --> 155         return self._data[key]
        156 


    KeyError: 'a'

    
    During handling of the above exception, another exception occurred:


    KeyError                                  Traceback (most recent call last)

    <ipython-input-17-5cfd0345d51c> in <module>
    ----> 1 print(gdf.query('strings == a'))
    

    /usr/local/lib/python3.8/dist-packages/cudf/core/dataframe.py in query(self, expr, local_dict)
       4172             }
       4173             # Run query
    -> 4174             boolmask = queryutils.query_execute(self, expr, callenv)
       4175             return self._apply_boolean_mask(boolmask)
       4176 


    /usr/local/lib/python3.8/dist-packages/cudf/utils/queryutils.py in query_execute(df, expr, callenv)
        212 
        213     # prepare col args
    --> 214     colarrays = [cudf.core.dataframe.extract_col(df, col) for col in columns]
        215 
        216     # wait to check the types until we know which cols are used


    /usr/local/lib/python3.8/dist-packages/cudf/utils/queryutils.py in <listcomp>(.0)
        212 
        213     # prepare col args
    --> 214     colarrays = [cudf.core.dataframe.extract_col(df, col) for col in columns]
        215 
        216     # wait to check the types until we know which cols are used


    /usr/local/lib/python3.8/dist-packages/cudf/core/dataframe.py in extract_col(df, col)
       7565         ):
       7566             return df.index._data.columns[0]
    -> 7567         return df.index._data[col]
       7568 
       7569 


    /usr/local/lib/python3.8/dist-packages/cudf/core/column_accessor.py in __getitem__(self, key)
        153 
        154     def __getitem__(self, key: Any) -> ColumnBase:
    --> 155         return self._data[key]
        156 
        157     def __setitem__(self, key: Any, value: Any):


    KeyError: 'a'


### Method 2:  Simple Columns

#### Filtering Strings by Column Values



```python
# Filtering based on the string column
print(gdf[gdf.strings == 'b'])
```

       integers strings
    1         2       b


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
