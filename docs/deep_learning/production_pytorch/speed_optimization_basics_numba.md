---
comments: true
---

# Speed Optimization Basics: Numba

## When to use Numba
- Numba works well when the code relies a lot on (1) numpy, (2) loops, and/or (2) cuda.
- Hence, we would like to maximize the use of numba in our code where possible where there are loops/numpy

## Numba CPU: nopython
- For a basic numba application, we can cecorate python function thus allowing it to run without python interpreter
- Essentially, it will compile the function with specific arguments once into machine code, then uses the cache subsequently

### With Numba: no python


```python
from numba import jit, prange
import numpy as np

# Numpy array of 10k elements
input_ndarray = np.random.rand(10000).reshape(10000)

# This is the only extra line of code you need to add
# which is a decorator
@jit(nopython=True)
def go_fast(a):
    trace = 0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i])
    return a + trace

%timeit go_fast(input_ndarray)
```

    161 µs ± 2.62 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)


### Without numba
- This is much slower, time measured in the millisecond space rather than microsecond with `@jit(nopython=True)` or `@njit`


```python
# Without numba: notice how this is really slow
def go_normal(a):
    trace = 0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i])
    return a + trace

%timeit go_normal(input_ndarray)
```

    10.5 ms ± 163 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)


## Numba CPU: parallel
- Here, instead of the normal `range()` function we would use for loops, we would need to use `prange()` which allows us to execute the loops in parallel on separate threads
- As you can see, it's slightly faster than `@jit(nopython=True)`


```python
@jit(nopython=True, parallel=True)
def go_even_faster(a):
    trace = 0
    for i in prange(a.shape[0]):
        trace += np.tanh(a[i])
    return a + trace

%timeit go_even_faster(input_ndarray)
```

    148 µs ± 71.3 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)


## Numba CPU: fastmath
- What if we relax our condition of strictly adhering to `IEEE 754`. 
- We can have faster performance (depends)
- I would say this is the least additional speed-up unless you really dig into areas where `fastmath=True` thrives


```python
@jit(nopython=True, parallel=True, fastmath=True)
def go_super_fast(a):
    trace = 0
    for i in prange(a.shape[0]):
        trace += np.tanh(a[i])
    return a + trace

%timeit go_super_fast(input_ndarray)
```

    113 µs ± 39.7 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)


## Summary
- When to use Numba
    - (1) numpy array or torch tensors,
    - (2) loops, and/or
    - (3) cuda
- Numba CPU: nopython¶
- Numba CPU: parallel 
- Numba CPU: fastmath
