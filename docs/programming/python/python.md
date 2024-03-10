---
comments: true
---

# Python
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ritchieng/deep-learning-wizard/blob/master/docs/programming/python/python.ipynb)

## Lists

### Creating List: Manual Fill


```python
lst = [0, 1, 2 ,3]
print(lst)
```

    [0, 1, 2, 3]


### Creating List: List Comprehension


```python
lst = [i for i in range(4)]
print(lst)
```

    [0, 1, 2, 3]


### Joining List with Blanks


```python
# To use .join(), your list needs to be of type string
lst_to_string = list(map(str, lst))

# Join the list of strings
lst_join = ' '.join(lst_to_string)
print(lst_join)
```

    0 1 2 3


### Joining List with Comma


```python
# Join the list of strings
lst_join = ', '.join(lst_to_string)
print(lst_join)
```

    0, 1, 2, 3


### Checking Lists Equal: Method 1
Returns `True` if equal, and `False` if unequal


```python
lst_unequal = [1, 1, 2, 3, 4, 4]
lst_equal = [0, 0, 0, 0, 0, 0]

print('-'*50)
print('Unequal List')
print('-'*50)

print(lst_unequal[1:])
print(lst_unequal[:-1])
bool_equal = lst_unequal[1:] == lst_unequal[:-1]
print(bool_equal)

print('-'*50)
print('Equal List')
print('-'*50)

print(lst_equal[1:])
print(lst_equal[:-1])
bool_equal = lst_equal[1:] == lst_equal[:-1]
print(bool_equal)
```

    --------------------------------------------------
    Unequal List
    --------------------------------------------------
    [1, 2, 3, 4, 4]
    [1, 1, 2, 3, 4]
    False
    --------------------------------------------------
    Equal List
    --------------------------------------------------
    [0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0]
    True


### Checking Lists Equal: Method 2
Returns `True` if equal, and `False` if unequal. Here, `all` essentially checks that there is no `False` in the list. 


```python
print('-'*50)
print('Unequal List')
print('-'*50)

lst_check = [i == lst_unequal[0] for i in lst_unequal]
bool_equal = all(lst_check)
print(bool_equal)

print('-'*50)
print('Equal List')
print('-'*50)

lst_check = [i == lst_equal[0] for i in lst_equal]
bool_equal = all(lst_check)
print(bool_equal)
```

    --------------------------------------------------
    Unequal List
    --------------------------------------------------
    False
    --------------------------------------------------
    Equal List
    --------------------------------------------------
    True


## Sets

### Removing Duplicate from List
Sets can be very useful for quickly removing duplicates from a list, essentially finding unique values


```python
lst_one = [1, 2, 3, 5]
lst_two = [1, 1, 2, 4]
lst_both = lst_one + lst_two
lst_no_duplicate = list(set(lst_both))

print(f'Original Combined List {lst_both}')
print(f'No Duplicated Combined List {lst_no_duplicate}')
```

    Original Combined List [1, 2, 3, 5, 1, 1, 2, 4]
    No Duplicated Combined List [1, 2, 3, 4, 5]


## Lambda, map, filter, reduce, partial

### Lambda
The syntax is simple `lambda your_variables: your_operation`

#### Add Function


```python
add = lambda x, y: x + y
add(2, 3)
```




    5



#### Multiply Function


```python
multiply = lambda x, y: x * y 
multiply(2, 3)
```




    6



### Map

#### Create List


```python
lst = [i for i in range(11)]
print(lst)
```

    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


#### Map Square Function to List


```python
square_element = map(lambda x: x**2, lst)

# This gives you a map object
print(square_element)

# You need to explicitly return a list
print(list(square_element))
```

    <map object at 0x7f08c8620438>
    [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100]


#### Create Multiple List


```python
lst_1 = [1, 2, 3, 4]
lst_2 = [2, 4, 6, 8]
lst_3 = [3, 6, 9, 12]
```

#### Map Add Function to Multiple Lists


```python
add_elements = map(lambda x, y, z : x + y + z, lst_1, lst_2, lst_3)
print(list(add_elements))
```

    [6, 12, 18, 24]


### Filter

#### Create List


```python
lst = [i for i in range(10)]
print(lst)
```

    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


#### Filter multiples of 3


```python
multiples_of_three = filter(lambda x: x % 3 == 0, lst)
print(list(multiples_of_three))
```

    [0, 3, 6, 9]


### Reduce
The syntax is `reduce(function, sequence)`. The function is applied to the elements in the list in a sequential manner. Meaning if `lst = [1, 2, 3, 4]` and you have a sum function, you would arrive with `((1+2) + 3) + 4`.


```python
from functools import reduce
sum_all = reduce(lambda x, y: x + y, lst)
# Here we've 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9
print(sum_all)
print(1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9)
```

    45
    45


### Partial
Allows us to predefine and freeze a function's argument. Combined with lambda, it allows us to have more flexibility beyond lambda's restriction of a single line.


```python
from functools import partial

def display_sum_three(a, b, c):
    sum_all = a + b + c
    print(f'Sum is {sum_all}')

fixed_args_func = partial(display_sum_three, b=3, c=4)

# Given fixed arguments b=3 and c=4
# We add the new variable against the fixed arguments
var_int = 1
fixed_args_func(var_int)

# More advanced mapping with partial
# Add a variable from 0 to 9 to the constants
print('-'*50)
_ = list(map(fixed_args_func, list(range(10))))

# How about using with lambda to modifying constants without
# declaring your function again?
print('-'*50)
_ = list(map(lambda x: fixed_args_func(x, b=2), list(range(10))))
```

    Sum is 8
    --------------------------------------------------
    Sum is 7
    Sum is 8
    Sum is 9
    Sum is 10
    Sum is 11
    Sum is 12
    Sum is 13
    Sum is 14
    Sum is 15
    Sum is 16
    --------------------------------------------------
    Sum is 6
    Sum is 7
    Sum is 8
    Sum is 9
    Sum is 10
    Sum is 11
    Sum is 12
    Sum is 13
    Sum is 14
    Sum is 15


## Generators
- Why: `generators` are typically more memory-efficient than using simple `for loops`
    - Imagine wanting to sum digits 0 to 1 trillion, using a list containing those numbers and summing them would be very RAM memory-inefficient.
    - Using a generator would allow you to sum one digit sequentially, staggering the RAM memory usage in steps.

- What: `generator` basically a function that returns an iterable object where we can iterate one bye one
- Types: generator functions and generator expressions
- Dependencies: we need to install a memory profiler, so install via `pip install memory_profiler`

### Simple custom generator function example: sum 1 to 1,000,000
- What: let's create a simple generator, allowing us to iterate through the digits 1 to 1,000,000 (inclusive) one by one with an increment of 1 at each step and summing them
- How: 2 step process with a `while` and a `yield`


```python
# Load memory profiler
%load_ext memory_profiler

# Here we take a step from 1
def create_numbers(end_number):
    current_number = 1
    
    # Step 1: while
    while current_number <= end_number:
        # Step 2: yield
        yield current_number
        
        # Add to current number
        current_number += 1
        
# Here we sum the digits 1 to 100 (inclusive) and time it
%memit total = sum(create_numbers(1e6))
print(total)
```

    peak memory: 46.50 MiB, increment: 0.28 MiB
    500000500000


#### Without generator function: sum with list
- Say we don't use a generator, and have a list of digits 0 to 1,000,000 (inclusive) in memory then sum them. 
- Notice how this is double the memory than using a generator!


```python
%memit total = sum(list(range(int(1e6) + 1)))
print(total)
```

    peak memory: 85.14 MiB, increment: 38.38 MiB
    500000500000


#### Without generator function: sum with for loop
- Say we don't use a generator and don't put all our numbers into a list 
- Notiice how this is much better than summing a list but still worst than a generator in terms of memory?


```python
def sum_with_loop(end_number):
    total = 0
    for i in range(end_number + 1):
        i += 1
        total += i
    
    return total

%memit total = sum_with_loop(int(1e6))
print(total)
```

    peak memory: 54.49 MiB, increment: 0.00 MiB
    500001500001


### Generator expression
- Like list/dictionary expressions, we can have generator expressions too
- We can quickly create generators this way, allowing us to make computations on the fly rather than pre-compute on a whole list/array of numbers
    - This is more memory efficient



```python
# Define the list
list_of_numbers = list(range(10))

# Find square root using the list comprehension
list_of_results = [number ** 2 for number in list_of_numbers]
print(list_of_results)

# Use generator expression to calculate the square root
generator_of_results = (number ** 2 for number in list_of_numbers)
print(generator_of_results)

for idx in range(10):
    print(next(generator_of_results))
```

    [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
    <generator object <genexpr> at 0x7f08c85aa4f8>
    0
    1
    4
    9
    16
    25
    36
    49
    64
    81


## Decorators
- This allows us to to modify our original function or even entirely replace it without changing the function's code. 
- It sounds mind-boggling, but a simple case I would like to illustrate here is using decorators for consistent logging (formatted print statements).
- For us to understand decorators, we'll first need to understand:
    - `first class objects`
    - `*args`
    - `*kwargs`

### First Class Objects


```python
def outer():
    def inner():
        print('Inside inner() function.')
        
    # This returns a function.
    return inner

# Here, we are assigning `outer()` function to the object `call_outer`.
call_outer = outer()

# Then we call `call_outer()` 
call_outer()
```

    Inside inner() function.


### *args
-  This is used to indicate that positional arguments should be stored in the variable args
- `*` is for iterables and positional parameters


```python
# Define dummy function
def dummy_func(*args):
    print(args)
    
# * allows us to extract positional variables from an iterable when we are calling a function
dummy_func(*range(10))

# If we do not use *, this would happen
dummy_func(range(10))

# See how we can have varying arguments?
dummy_func(*range(2))
```

    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    (range(0, 10),)
    (0, 1)


### **kwargs
- `**` is for dictionaries & key/value pairs


```python
# New dummy function
def dummy_func_new(**kwargs):
    print(kwargs)
    
# Call function with no arguments
dummy_func_new()

# Call function with 2 arguments
dummy_func_new(a=0, b=1)

# Again, there's no limit to the number of arguments.
dummy_func_new(a=0, b=1, c=2)

# Or we can just pass the whole dictionary object if we want
new_dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
dummy_func_new(**new_dict)
```

    {}
    {'a': 0, 'b': 1}
    {'a': 0, 'b': 1, 'c': 2}
    {'a': 0, 'b': 1, 'c': 2, 'd': 3}


### Decorators as Logger and Debugging
- A simple way to remember the power of decorators is that the decorator (the nested function illustrated below) can
    - (1) access the passed arguments of the decorated function and
    - (2) access the decorated function
- Therefore this allows us to modify the decorated function without changing the decorated function


```python
# Create a nested function that will be our decorator
def function_inspector(func):
    def inner(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f'Function args: {args}')
        print(f'Function kwargs: {kwargs}')
        print(f'Function return result: {result}')
        return result
    return inner

# Decorate our multiply function with our logger for easy logging
# Of arguments pass to the function and results returned
@function_inspector
def multiply_func(num_one, num_two):
    return num_one * num_two

multiply_result = multiply_func(num_one=1, num_two=2)
```

    Function args: ()
    Function kwargs: {'num_one': 1, 'num_two': 2}
    Function return result: 2


## Dates

### Get Current Date


```python
import datetime
now = datetime.datetime.now()
print(now)
```

    2019-08-12 14:20:45.604849


### Get Clean String Current Date


```python
# YYYY-MM-DD
now.date().strftime('20%y-%m-%d')
```




    '2019-08-12'



### Count Business Days


```python
# Number of business days in a month from Jan 2019 to Feb 2019
import numpy as np
days = np.busday_count('2019-01', '2019-02')
print(days)
```

    23


## Progress Bars

### TQDM
Simple progress bar via `pip install tqdm`


```python
from tqdm import tqdm
import time
for i in tqdm(range(100)):
    time.sleep(0.1)
    pass
```

    100%|██████████| 100/100 [00:10<00:00,  9.91it/s]


## Check Paths

### Check Path Exists
- Check if directory exists


```python
import os
directory='new_dir'
print(os.path.exists(directory))

# Magic function to list all folders
!ls -d */
```

    False
    ls: cannot access '*/': No such file or directory


### Check Path Exists Otherwise Create Folder
- Check if directory exists, otherwise make folder


```python
if not os.path.exists(directory):
    os.makedirs(directory)
    
# Magic function to list all folders
!ls -d */

# Remove directory
!rmdir new_dir
```

    new_dir/


## Exception Handling

### Try, Except, Finally: Error
- This is very handy and often exploited to patch up (save) poorly written code
- You can use general exceptions or specific ones like `ValueError`, `KeyboardInterrupt` and `MemoryError` to name a few


```python
value_one = 'a'
value_two = 2

# Try the following line of code
try:
    final_sum = value_one / value_two
    print('Code passed!')
# If the code above fails, code nested under except will be executed
except:
    print('Code failed!')
# This will run no matter whether the nested code in try or except is executed
finally:
    print('Ran code block regardless of error or not.')
```

    Code failed!
    Ran code block regardless of error or not.


### Try, Except, Finally: No Error
- There won't be errors because you can divide 4 with 2


```python
value_one = 4
value_two = 2

# Try the following line of code
try:
    final_sum = value_one / value_two
    print('Code passed!')
# If the code above fails, code nested under except will be executed
except:
    print('Code failed!')
# This will run no matter whether the nested code in try or except is executed
finally:
    print('Ran code block regardless of error or not.')
```

    Code passed!
    Ran code block regardless of error or not.


### Assertion
- This comes in handy when you want to enforce strict requirmenets of a certain value, shape, value type, or others


```python
for i in range(10):
    assert i <= 5, 'Value is more than 5, rejected'
    print(f'Passed assertion for value {i}')
```

    Passed assertion for value 0
    Passed assertion for value 1
    Passed assertion for value 2
    Passed assertion for value 3
    Passed assertion for value 4
    Passed assertion for value 5



    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-2-d9d077e139a9> in <module>
          1 for i in range(10):
    ----> 2     assert i <= 5, 'Value is more than 5, rejected'
          3     print(f'Passed assertion for value {i}')


    AssertionError: Value is more than 5, rejected


## Asynchronous

### Concurrency, Parallelism, Asynchronous
- Concurrency (single CPU core): multiple threads on a single core running in **sequence**, only 1 thread is making progress at any point
    - Think of 1 human, packing a box then wrapping the box
- Parallelism (mutliple GPU cores): multiple threads on multiple cores running in **parallel**, multiple threads can be making progress
    - Think of 2 humans, one packing a box, another wrapping the box
- Asynchronous: concurrency but with a more dynamic system that moves amongst threads more efficiently rather than waiting for a task to finish then moving to the next task
    - Python's `asyncio` allows us to code asynchronously
    - Benefits:
        - Scales better if you need to wait on a lot of processes
            - Less memory (easier in this sense) to wait on thousands of co-routines than running on thousands of threads
        - Good for IO bound uses like reading/saving from databases while subsequently running other computation
        - Easier management than multi-thread processing like in parallel programming
            - In the sense that everything operates sequentially in the same memory space

### Asynchronous Key Components
- The three main parts are (1) coroutines and subroutines, (2) event loops, and (3) future.
    - Co-routine and subroutines
        - Subroutine: the usual function
        - Coroutine: this allows us to maintain states with memory of where things stopped so we can swap amongst subroutines
            - `async` declares a function as a coroutine
            - `await` to call a coroutine
    - Event loops
    - Future

### Synchronous 2 Function Calls


```python
import timeit
def add_numbers(num_1, num_2):
    print('Adding')
    time.sleep(1)
    return num_1 + num_2

def display_sum(num_1, num_2):
    total_sum = add_numbers(num_1, num_2)
    print(f'Total sum {total_sum}')
    
def main():
    display_sum(2, 2)
    display_sum(2, 2)

start = timeit.default_timer()

main()

end = timeit.default_timer()
total_time = end - start

print(f'Total time {total_time:.2f}s')
```

    Adding
    Total sum 4
    Adding
    Total sum 4
    Total time 2.00s


### Parallel 2 Function Calls


```python
from multiprocessing import Pool
from functools import partial

start = timeit.default_timer()

pool = Pool()
result = pool.map(partial(display_sum, num_2=2), [2, 2]) 

end = timeit.default_timer()
total_time = end - start

print(f'Total time {total_time:.2f}s')
```

    Adding
    Adding
    Total sum 4
    Total sum 4
    Total time 1.08s


### Asynchronous 2 Function Calls
For this use case, it'll take half the time compared to a synchronous application and slightly faster than parallel application (although not always true for parallel except in this case)


```python
import asyncio
import timeit
import time
    
async def add_numbers(num_1, num_2):
    print('Adding')
    await asyncio.sleep(1)
    return num_1 + num_2 

async def display_sum(num_1, num_2):
    total_sum = await add_numbers(num_1, num_2)
    print(f'Total sum {total_sum}')
    
async def main():
    # .gather allows us to group subroutines
    await asyncio.gather(display_sum(2, 2), 
                         display_sum(2, 2))
    
start = timeit.default_timer()

# For .ipynb, event loop already done
await main()

# For .py
# asyncio.run(main())

end = timeit.default_timer()
total_time = end - start

print(f'Total time {total_time:.4f}s')
```

    Adding
    Adding
    Total sum 4
    Total sum 4
    Total time 1.0021s

