# PyTorch Fundamentals - Matrices 

## 1. Matrices


### 1.0 Matrices Brief Introduction
* [x] Basic definition: rectangular array of numbers.
* [x] Tensors (PyTorch)
* [x] Ndarrays (NumPy)

**2 x 2 Matrix (R x C)**

1 | 1 
--- | ---
1 | 1 

**2 x 3 Matrix**

1 | 1 | 1
--- | ---| ---
1 | 1 | 1 

### 1.1 Creating Matrices


```python
import numpy as np
```


```python
# Creating a 2x2 array
arr = [[1, 2], [3, 4]]
print(arr)
```

```python tab="Output"
[[1, 2], [3, 4]]
```


```python
# Convert to NumPy
np.array(arr)
```


```python tab="Output"

array([[1, 2],
       [3, 4]])

```


```python
import torch
```


```python
# Convert to PyTorch Tensor
torch.Tensor(arr)
```

```python tab="Output"
1  2
3  4
[torch.FloatTensor of size 2x2]

```
    
### 1.2 Create Matrices with Default Values


```python
np.ones((2, 2))
```

```python tab="Output"
array([[ 1.,  1.],
       [ 1.,  1.]])

```







```python
torch.ones((2, 2))
```




    
     1  1
     1  1
    [torch.FloatTensor of size 2x2]




```python
np.random.rand(2, 2)
```




    array([[ 0.68270631,  0.87721678],
           [ 0.07420986,  0.79669375]])




```python
torch.rand(2, 2)
```




    
     0.3900  0.8268
     0.3888  0.5914
    [torch.FloatTensor of size 2x2]



### 1.3 Seeds for Reproducibility


```python
# Seed
np.random.seed(0)
np.random.rand(2, 2)
```




    array([[ 0.5488135 ,  0.71518937],
           [ 0.60276338,  0.54488318]])




```python
# Seed
np.random.seed(0)
np.random.rand(2, 2)
```




    array([[ 0.5488135 ,  0.71518937],
           [ 0.60276338,  0.54488318]])




```python
# No seed
np.random.rand(2, 2)
```




    array([[ 0.56804456,  0.92559664],
           [ 0.07103606,  0.0871293 ]])




```python
# No seed
np.random.rand(2, 2)
```




    array([[ 0.0202184 ,  0.83261985],
           [ 0.77815675,  0.87001215]])




```python
# Torch Seed
torch.manual_seed(0)
torch.rand(2, 2)
```




    
     0.5488  0.5928
     0.7152  0.8443
    [torch.FloatTensor of size 2x2]




```python
# Torch Seed
torch.manual_seed(0)
torch.rand(2, 2)
```




    
     0.5488  0.5928
     0.7152  0.8443
    [torch.FloatTensor of size 2x2]




```python
# Torch No Seed
torch.rand(2, 2)
```




    
     0.6028  0.8579
     0.5449  0.8473
    [torch.FloatTensor of size 2x2]




```python
# Torch No Seed
torch.rand(2, 2)
```




    
     0.4237  0.6236
     0.6459  0.3844
    [torch.FloatTensor of size 2x2]



**Seed for GPU is different for now...**


```python
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
```

### 1.3 NumPy and Torch Bridge

### NumPy to Torch 


```python
# Numpy array
np_array = np.ones((2, 2))
```


```python
print(np_array)
```

    [[ 1.  1.]
     [ 1.  1.]]



```python
print(type(np_array))
```

    <class 'numpy.ndarray'>



```python
# Convert to Torch Tensor
torch_tensor = torch.from_numpy(np_array)
```


```python
print(torch_tensor)
```

    
     1  1
     1  1
    [torch.DoubleTensor of size 2x2]
    



```python
print(type(torch_tensor))
```

    <class 'torch.DoubleTensor'>



```python
# Data types matter: intentional error
np_array_new = np.ones((2, 2), dtype=np.int8)
torch.from_numpy(np_array_new)
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-57-b8b085f9b39d> in <module>()
          1 # Data types matter
          2 np_array_new = np.ones((2, 2), dtype=np.int8)
    ----> 3 torch.from_numpy(np_array_new)
    

    RuntimeError: can't convert a given np.ndarray to a tensor - it has an invalid type. The only supported types are: double, float, int64, int32, and uint8.


**The conversion supports:**
1. `double`
2. `float` 
3. `int64`, `int32`, `uint8` 


```python
# Data types matter
np_array_new = np.ones((2, 2), dtype=np.int64)
torch.from_numpy(np_array_new)
```




    
     1  1
     1  1
    [torch.LongTensor of size 2x2]




```python
# Data types matter
np_array_new = np.ones((2, 2), dtype=np.int32)
torch.from_numpy(np_array_new)
```




    
     1  1
     1  1
    [torch.IntTensor of size 2x2]




```python
# Data types matter
np_array_new = np.ones((2, 2), dtype=np.uint8)
torch.from_numpy(np_array_new)
```




    
     1  1
     1  1
    [torch.ByteTensor of size 2x2]




```python
# Data types matter
np_array_new = np.ones((2, 2), dtype=np.float64)
torch.from_numpy(np_array_new)
```




    
     1  1
     1  1
    [torch.DoubleTensor of size 2x2]




```python
# Data types matter
np_array_new = np.ones((2, 2), dtype=np.float32)
torch.from_numpy(np_array_new)
```




    
     1  1
     1  1
    [torch.FloatTensor of size 2x2]




```python
# Data types matter
np_array_new = np.ones((2, 2), dtype=np.double)
torch.from_numpy(np_array_new)
```




    
     1  1
     1  1
    [torch.DoubleTensor of size 2x2]



**Summary**
<br />These things don't matter much now. But later when you see error messages that require these particular tensor types, refer to this guide!

| NumPy Array Type        | Torch Tensor Type           |
| :-------------: |:--------------:|
| int64     | LongTensor |
| int32     | IntegerTensor |
| uint8      | ByteTensor      |
| float64 | DoubleTensor     |
| float32 | FloatTensor      |
| double | DoubleTensor      |

### Torch to NumPy


```python
torch_tensor = torch.ones(2, 2)
```


```python
type(torch_tensor)
```




    torch.FloatTensor




```python
torch_to_numpy = torch_tensor.numpy()
```


```python
type(torch_to_numpy)
```




    numpy.ndarray



### 1.4 Tensors on CPU vs GPU


```python
# CPU
tensor_cpu = torch.ones(2, 2)
```


```python
# CPU to GPU
if torch.cuda.is_available():
    tensor_cpu.cuda()
```


```python
# GPU to CPU
tensor_cpu.cpu()
```

### 1.5 Tensor Operations

####  Resizing Tensor


```python
a = torch.ones(2, 2)
print(a)
```

    
     1  1
     1  1
    [torch.FloatTensor of size 2x2]
    



```python
print(a.size())
```

    torch.Size([2, 2])



```python
a.view(4)
```




    
     1
     1
     1
     1
    [torch.FloatTensor of size 4]




```python
a.view(4).size()
```




    torch.Size([4])



### Element-wise Addition


```python
a = torch.ones(2, 2)
print(a)
```

    
     1  1
     1  1
    [torch.FloatTensor of size 2x2]
    



```python
b = torch.ones(2, 2)
print(b)
```

    
     1  1
     1  1
    [torch.FloatTensor of size 2x2]
    



```python
# Element-wise addition
c = a + b
print(c)
```

    
     2  2
     2  2
    [torch.FloatTensor of size 2x2]
    



```python
# Element-wise addition
c = torch.add(a, b)
print(c)
```

    
     2  2
     2  2
    [torch.FloatTensor of size 2x2]
    



```python
# In-place addition
print('Old c tensor')
print(c)

c.add_(a)

print('-'*60)
print('New c tensor')
print(c)
```

    Old c tensor
    
     2  2
     2  2
    [torch.FloatTensor of size 2x2]
    
    ------------------------------------------------------------
    New c tensor
    
     3  3
     3  3
    [torch.FloatTensor of size 2x2]
    


#### Element-wise Subtraction


```python
print(a)
print(b)
```

    
     1  1
     1  1
    [torch.FloatTensor of size 2x2]
    
    
     1  1
     1  1
    [torch.FloatTensor of size 2x2]
    



```python
a - b
```




    
     0  0
     0  0
    [torch.FloatTensor of size 2x2]




```python
# Not in-place
print(a.sub(b))
print(a)
```

    
     0  0
     0  0
    [torch.FloatTensor of size 2x2]
    
    
     1  1
     1  1
    [torch.FloatTensor of size 2x2]
    



```python
# Inplace
print(a.sub_(b))
print(a)
```

    
     0  0
     0  0
    [torch.FloatTensor of size 2x2]
    
    
     0  0
     0  0
    [torch.FloatTensor of size 2x2]
    


### Element-Wise Multiplication


```python
a = torch.ones(2, 2)
print(a)
b = torch.zeros(2, 2)
print(b)
```

    
     1  1
     1  1
    [torch.FloatTensor of size 2x2]
    
    
     0  0
     0  0
    [torch.FloatTensor of size 2x2]
    



```python
a * b
```




    
     0  0
     0  0
    [torch.FloatTensor of size 2x2]




```python
# Not in-place
print(torch.mul(a, b))
print(a)
```

    
     0  0
     0  0
    [torch.FloatTensor of size 2x2]
    
    
     1  1
     1  1
    [torch.FloatTensor of size 2x2]
    



```python
# In-place
print(a.mul_(b))
print(a)
```

    
     0  0
     0  0
    [torch.FloatTensor of size 2x2]
    
    
     0  0
     0  0
    [torch.FloatTensor of size 2x2]
    


### Element-Wise Division


```python
a = torch.ones(2, 2)
b = torch.zeros(2, 2)
```


```python
b / a
```




    
     0  0
     0  0
    [torch.FloatTensor of size 2x2]




```python
torch.div(b, a)
```




    
     0  0
     0  0
    [torch.FloatTensor of size 2x2]




```python
# Inplace
b.div_(a)
```




    
     0  0
     0  0
    [torch.FloatTensor of size 2x2]



### Tensor Mean

$$1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 = 55$$

$$ mean = 55 /10 = 5.5 $$



```python
a = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
a.size()
```




    torch.Size([10])




```python
a.mean(dim=0)
```




    
     5.5000
    [torch.FloatTensor of size 1]




```python
a.mean(dim=1)
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-7-81aec0cf1c00> in <module>()
    ----> 1 a.mean(dim=1)
    

    RuntimeError: dimension out of range (expected to be in range of [-1, 0], but got 1)



```python
a = torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
```


```python
a.size()
```




    torch.Size([2, 10])




```python
a.mean(dim=1)
```




    
     5.5000
     5.5000
    [torch.FloatTensor of size 2x1]



## Tensor Standard Deviation


```python
a = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
a.std(dim=0)
```




    
     3.0277
    [torch.FloatTensor of size 1]



---
## We've learnt to...
!!! success
    * [x] Create Matrices
    * [x] Create Matrices with Default Initialization Values
        * [x] Zeros 
        * [x] Ones
    * [x] Initialize Seeds for Reproducibility on GPU and CPU
    * [x] Convert Matrices: NumPy :arrow_right: Torch and Torch :arrow_right: NumPy
    * [x] Move Tensors: CPU :arrow_right: GPU and GPU :arrow_right: CPU
    * [x] Run Important Tensor Operations
        * [x] Element-wise addition, subtraction, multiplication and division
        * [x] Resize
        * [x] Calculate mean 
        * [x] Calculate standard deviation
