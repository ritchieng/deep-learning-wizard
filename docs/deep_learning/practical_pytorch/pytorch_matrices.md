---
comments: true
---

# Matrices with PyTorch 

!!! tip "Run Jupyter Notebook"
    You can run the code for this section in this [jupyter notebook link](https://github.com/ritchieng/deep-learning-wizard/blob/master/docs/deep_learning/practical_pytorch/pytorch_matrices.ipynb).
    
## Matrices

### Matrices Brief Introduction
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

### Creating Matrices


!!! note "Create list"
    ```python
    # Creating a 2x2 array
    arr = [[1, 2], [3, 4]]
    print(arr)
    ```

```python
[[1, 2], [3, 4]]
```

!!! note "Create numpy array via list"
    ```python
    import numpy as np
    ```
    ```python
    # Convert to NumPy
    np.array(arr)
    ```

```python

array([[1, 2],
       [3, 4]])

```

!!! note "Convert numpy array to PyTorch tensor"
    ```python
    import torch
    ```
    
    
    ```python
    # Convert to PyTorch Tensor
    torch.Tensor(arr)
    ```

```python
1  2
3  4
[torch.FloatTensor of size 2x2]

```
    
### Create Matrices with Default Values


!!! note "Create 2x2 numpy array of 1's"
    ```python
    np.ones((2, 2))
    ```

```python
array([[ 1.,  1.],
       [ 1.,  1.]])

```

!!! note "Create 2x2 torch tensor of 1's"

    ```python
    torch.ones((2, 2))
    ```

```
 1  1
 1  1
[torch.FloatTensor of size 2x2]
```


!!! note "Create 2x2 numpy array of random numbers"
    ```python
    np.random.rand(2, 2)
    ```

```python
array([[ 0.68270631,  0.87721678],
       [ 0.07420986,  0.79669375]])

```

!!! note "Create 2x2 PyTorch tensor of random numbers"

    ```python
    torch.rand(2, 2)
    ```


```python
0.3900  0.8268
0.3888  0.5914
[torch.FloatTensor of size 2x2]

```

### Seeds for Reproducibility

!!! question "Why do we need seeds?"
    We need seeds to enable reproduction of experimental results. This becomes critical later on where you can easily let people reproduce your code's output exactly as you've produced.
    
!!! note "Create seed to enable fixed numbers for random number generation "
    ```python
    # Seed
    np.random.seed(0)
    np.random.rand(2, 2)
    ```


```python
array([[ 0.5488135 ,  0.71518937],
       [ 0.60276338,  0.54488318]])

```

!!! note "Repeat random array generation to check"
    If you do not set the seed, you would not get the same set of numbers like here.
    ```python
    # Seed
    np.random.seed(0)
    np.random.rand(2, 2)
    ```


```python
array([[ 0.5488135 ,  0.71518937],
       [ 0.60276338,  0.54488318]])

```


!!! note "Create a numpy array without seed"
    Notice how you get different numbers compared to the first 2 tries?
    ```python
    # No seed
    np.random.rand(2, 2)
    ```


```python
array([[ 0.56804456,  0.92559664],
       [ 0.07103606,  0.0871293 ]])

```

    
!!! note "Repeat numpy array generation without seed"
    You get the point now, you get a totally different set of numbers.
    ```python
    # No seed
    np.random.rand(2, 2)
    ```


```python
array([[ 0.0202184 ,  0.83261985],
       [ 0.77815675,  0.87001215]])
```

    
!!! note "Create a PyTorch tensor with a fixed seed"
    ```python
    # Torch Seed
    torch.manual_seed(0)
    torch.rand(2, 2)
    ```
    
    
```python
0.5488  0.5928
0.7152  0.8443
[torch.FloatTensor of size 2x2]

```
!!! note "Repeat creating a PyTorch fixed seed tensor"
    ```python
    # Torch Seed
    torch.manual_seed(0)
    torch.rand(2, 2)
    ```
    
```python
0.5488  0.5928
0.7152  0.8443
[torch.FloatTensor of size 2x2]

```



!!! note "Creating a PyTorch tensor without seed"
    Like with a numpy array of random numbers without seed, you will not get the same results as above.
    ```python
    # Torch No Seed
    torch.rand(2, 2)
    ```

```python
0.6028  0.8579
0.5449  0.8473
[torch.FloatTensor of size 2x2]
```   


!!! note "Repeat creating a PyTorch tensor without seed"
    Notice how these are different numbers again?
    ```python
    # Torch No Seed
    torch.rand(2, 2)
    ```


```python
0.4237  0.6236
0.6459  0.3844
[torch.FloatTensor of size 2x2]

```

**Seed for GPU is different for now...**

!!! note "Fix a seed for GPU tensors"
    When you conduct deep learning experiments, typically you want to use GPUs to accelerate your computations and fixing seed for tensors on GPUs is different from CPUs as we have done above. 
    ```python
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    ```

### NumPy and Torch Bridge

#### NumPy to Torch 


!!! note "Create a numpy array of 1's"
    ```python
    # Numpy array
    np_array = np.ones((2, 2))
    ```
    
    ```python
    print(np_array)
    ```

```python
[[ 1.  1.]
[ 1.  1.]]

```
    
!!! note "Get the type of class for the numpy array"

    ```python
    print(type(np_array))
    ```

```python
<class 'numpy.ndarray'>

```
    


!!! note "Convert numpy array to PyTorch tensor"

    ```python
    # Convert to Torch Tensor
    torch_tensor = torch.from_numpy(np_array)
    ```
    
    
    ```python
    print(torch_tensor)
    ```

```python
 1  1
 1  1
[torch.DoubleTensor of size 2x2]
```

!!! note "Get type of class for PyTorch tensor"
    Notice how it shows it's a torch DoubleTensor? There're actually tensor types and it depends on the numpy data type.
    ```python
    print(type(torch_tensor))
    ```

```python
<class 'torch.DoubleTensor'>
```
    


!!! note "Create PyTorch tensor from a different numpy datatype"
    You will get an error running this code because PyTorch tensor don't support all datatype. 
    ```python
    # Data types matter: intentional error
    np_array_new = np.ones((2, 2), dtype=np.int8)
    torch.from_numpy(np_array_new)
    ```

```python
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)

<ipython-input-57-b8b085f9b39d> in <module>()
      1 # Data types matter
      2 np_array_new = np.ones((2, 2), dtype=np.int8)
----> 3 torch.from_numpy(np_array_new)


RuntimeError: can't convert a given np.ndarray to a tensor - it has an invalid type. The only supported types are: double, float, int64, int32, and uint8.


```

!!! help "What conversion support does Numpy to PyTorch tensor bridge gives?"
    - `double`
    - `float` 
    - `int64`, `int32`, `uint8` 


!!! note "Create PyTorch long tensor"
    See how a int64 numpy array gives you a PyTorch long tensor?
    ```python
    # Data types matter
    np_array_new = np.ones((2, 2), dtype=np.int64)
    torch.from_numpy(np_array_new)
    ```


```
1  1
1  1
[torch.LongTensor of size 2x2]

```

!!! note "Create PyTorch int tensor"
    
    ```python
    # Data types matter
    np_array_new = np.ones((2, 2), dtype=np.int32)
    torch.from_numpy(np_array_new)
    ```


```python
1  1
1  1
[torch.IntTensor of size 2x2]
```

!!! note "Create PyTorch byte tensor"

    ```python
    # Data types matter
    np_array_new = np.ones((2, 2), dtype=np.uint8)
    torch.from_numpy(np_array_new)
    ```


```python
1  1
1  1
[torch.ByteTensor of size 2x2]

```

    
!!! note "Create PyTorch Double Tensor"

    ```python
    # Data types matter
    np_array_new = np.ones((2, 2), dtype=np.float64)
    torch.from_numpy(np_array_new)
    ```
    
    Alternatively you can do this too via `np.double`
    
    ```python
    # Data types matter
    np_array_new = np.ones((2, 2), dtype=np.double)
    torch.from_numpy(np_array_new)
    ```

```python
1  1
1  1
[torch.DoubleTensor of size 2x2]

```
    
!!! note "Create PyTorch Float Tensor"     
    ```python
    # Data types matter
    np_array_new = np.ones((2, 2), dtype=np.float32)
    torch.from_numpy(np_array_new)
    ```



```python

1  1
1  1
[torch.FloatTensor of size 2x2]

```


**Summary**
!!! bug "Tensor Type Bug Guide"
    These things don't matter much now. But later when you see error messages that require these particular tensor types, refer to this guide!

| NumPy Array Type        | Torch Tensor Type           |
| :-------------: |:--------------:|
| int64     | LongTensor |
| int32     | IntegerTensor |
| uint8      | ByteTensor      |
| float64 | DoubleTensor     |
| float32 | FloatTensor      |
| double | DoubleTensor      |

#### Torch to NumPy

!!! note "Create PyTorch tensor of 1's"
    You would realize this defaults to a float tensor by default if you do this.

    ```python
    torch_tensor = torch.ones(2, 2)
    ```


    ```python
    type(torch_tensor)
    ```


```python
torch.FloatTensor
```

    
!!! note "Convert tensor to numpy"
    It's as simple as this.
    
    ```python
    torch_to_numpy = torch_tensor.numpy()
    ```


    ```python
    type(torch_to_numpy)
    ```



```python
# Wowza, we did it.
numpy.ndarray
```

### Tensors on CPU vs GPU

!!! note "Move tensor to CPU and back"
    This by default creates a tensor on CPU. You do not need to do anything.
    ```python
    # CPU
    tensor_cpu = torch.ones(2, 2)
    ```
    
    If you would like to send a tensor to your GPU, you just need to do a simple `.cuda()`
    
    ```python
    # CPU to GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tensor_cpu.to(device)
    ```
    
    And if you want to move that tensor on the GPU back to the CPU, just do the following.
    
    ```python
    # GPU to CPU
    tensor_cpu.cpu()
    ```

### Tensor Operations

####  Resizing Tensor

!!! note "Creating a 2x2 tensor"
    ```python
    a = torch.ones(2, 2)
    print(a)
    ```

```python
1  1
1  1
[torch.FloatTensor of size 2x2]

```

!!! note "Getting size of tensor"
    ```python
    print(a.size())
    ```

```python
torch.Size([2, 2])
```

!!! note "Resize tensor to 4x1"
    ```python
    a.view(4)
    ```



```python
1
1
1
1
[torch.FloatTensor of size 4]
```
    



!!! note "Get size of resized tensor"
    ```python
    a.view(4).size()
    ```
```python
torch.Size([4])
```

#### Element-wise Addition


!!! note "Creating first 2x2 tensor"
    ```python
    a = torch.ones(2, 2)
    print(a)
    ```

```python
1  1
1  1
[torch.FloatTensor of size 2x2]

```

!!! note "Creating second 2x2 tensor"
    ```python
    b = torch.ones(2, 2)
    print(b)
    ```

```python
1  1
1  1
[torch.FloatTensor of size 2x2]

```


!!! note "Element-wise addition of 2 tensors"
    ```python
    # Element-wise addition
    c = a + b
    print(c)
    ```

```python
 2  2
 2  2
[torch.FloatTensor of size 2x2]

```
    

   
!!! note "Alternative element-wise addition of 2 tensors"
    ```python
    # Element-wise addition
    c = torch.add(a, b)
    print(c)
    ```

```python
 2  2
 2  2
[torch.FloatTensor of size 2x2]

```


!!! note "In-place element-wise addition"
    This would replace the c tensor values with the new addition. 
    
    ```python
    # In-place addition
    print('Old c tensor')
    print(c)
    
    c.add_(a)
    
    print('-'*60)
    print('New c tensor')
    print(c)
    ```

```python
Old c tensor

 2  2
 2  2
[torch.FloatTensor of size 2x2]

------------------------------------------------------------
New c tensor

 3  3
 3  3
[torch.FloatTensor of size 2x2]

```



#### Element-wise Subtraction

!!! note "Check values of tensor a and b'"
    Take note that you've created tensor a and b of sizes 2x2 filled with 1's each above. 
    ```python
    print(a)
    print(b)
    ```

```python
 1  1
 1  1
[torch.FloatTensor of size 2x2]


 1  1
 1  1
[torch.FloatTensor of size 2x2]
```


!!! note "Element-wise subtraction: method 1"
    ```python
    a - b
    ```


```python
0  0
0  0
[torch.FloatTensor of size 2x2]

```

!!! note "Element-wise subtraction: method 2"
    ```python
    # Not in-place
    print(a.sub(b))
    print(a)
    ```

    
```python
0  0
0  0
[torch.FloatTensor of size 2x2]


1  1
1  1
[torch.FloatTensor of size 2x2]
```
    

!!! note "Element-wise subtraction: method 3"
    This will replace a with the final result filled with 2's
    ```python
    # Inplace
    print(a.sub_(b))
    print(a)
    ```
    
```python
0  0
0  0
[torch.FloatTensor of size 2x2]


0  0
0  0
[torch.FloatTensor of size 2x2]
```
   
#### Element-Wise Multiplication

!!! note "Create tensor a and b of sizes 2x2 filled with 1's and 0's"
    ```python
    a = torch.ones(2, 2)
    print(a)
    b = torch.zeros(2, 2)
    print(b)
    ```

    
```python
1  1
1  1
[torch.FloatTensor of size 2x2]


0  0
0  0
[torch.FloatTensor of size 2x2]

```

!!! note "Element-wise multiplication: method 1"
    ```python
    a * b
    ```


```python
0  0
0  0
[torch.FloatTensor of size 2x2]
```

    

!!! note "Element-wise multiplication: method 2"
    ```python
    # Not in-place
    print(torch.mul(a, b))
    print(a)
    ```

```python
0  0
0  0
[torch.FloatTensor of size 2x2]

1  1
1  1
[torch.FloatTensor of size 2x2]
```

!!! note "Element-wise multiplication: method 3"
    ```python
    # In-place
    print(a.mul_(b))
    print(a)
    ```

    
```python
0  0
0  0
[torch.FloatTensor of size 2x2]

0  0
0  0
[torch.FloatTensor of size 2x2]
```


#### Element-Wise Division


!!! note "Create tensor a and b of sizes 2x2 filled with 1's and 0's"
    ```python
    a = torch.ones(2, 2)
    print(a)
    b = torch.zeros(2, 2)
    print(b)
    ```

    
```python
1  1
1  1
[torch.FloatTensor of size 2x2]


0  0
0  0
[torch.FloatTensor of size 2x2]

```

!!! note "Element-wise division: method 1"
    ```python
    b / a
    ```


```python
0  0
0  0
[torch.FloatTensor of size 2x2]

```

!!! note "Element-wise division: method 2"
    ```python
    torch.div(b, a)
    ```

```python
0  0
0  0
[torch.FloatTensor of size 2x2]

```

!!! note "Element-wise division: method 3"
    ```python
    # Inplace
    b.div_(a)
    ```

```python
0  0
0  0
[torch.FloatTensor of size 2x2]

```

#### Tensor Mean

$$1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 = 55$$

$$ mean = 55 /10 = 5.5 $$


!!! note "Create tensor of size 10 filled from 1 to 10"
    ```python
    a = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    a.size()
    ```


```python
torch.Size([10])

```


!!! note "Get tensor mean"
    Here we get 5.5 as we've calculated manually above.
    
    ```python
    a.mean(dim=0)
    ```

```python
    
5.5000
[torch.FloatTensor of size 1]

```



!!! note "Get tensor mean on second dimension"
    Here we get an error because the tensor is of size 10 and not 10x1 so there's no second dimension to calculate.
    
    ```python
    a.mean(dim=1)
    ```

```python

RuntimeError                              Traceback (most recent call last)

<ipython-input-7-81aec0cf1c00> in <module>()
----> 1 a.mean(dim=1)


RuntimeError: dimension out of range (expected to be in range of [-1, 0], but got 1)

```

!!! note "Create a 2x10 Tensor, of 1-10 digits each"
    
    ```python
    a = torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    ```
    ```python
    a.size()
    ```

```python
torch.Size([2, 10])
```

    

!!! note "Get tensor mean on second dimension"
    Here we won't get an error like previously because we've a tensor of size 2x10

    ```python
    a.mean(dim=1)
    ```


```python
 5.5000
 5.5000
[torch.FloatTensor of size 2x1]

```





#### Tensor Standard Deviation

!!! note "Get standard deviation of tensor"
    ```python
    
    a = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    a.std(dim=0)
    ```


```python
 3.0277
[torch.FloatTensor of size 1]

```

## Summary
We've learnt to...

!!! success
    * [x] Create Matrices
    * [x] Create Matrices with Default Initialization Values
        * [x] Zeros 
        * [x] Ones
    * [x] Initialize Seeds for Reproducibility on GPU and CPU
    * [x] Convert Matrices: NumPy to Torch and Torch to NumPy
    * [x] Move Tensors: CPU to GPU and GPU to CPU
    * [x] Run Important Tensor Operations
        * [x] Element-wise addition, subtraction, multiplication and division
        * [x] Resize
        * [x] Calculate mean 
        * [x] Calculate standard deviation

## Citation
If you have found these useful in your research, presentations, school work, projects or workshops, feel free to cite using this DOI.

[![DOI](https://zenodo.org/badge/139945544.svg)](https://zenodo.org/badge/latestdoi/139945544)