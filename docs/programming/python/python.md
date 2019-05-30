
# Python

## Lambda, map, filter, reduce

### Lambda

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

    <map object at 0x7f84dc5aaa90>
    [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100]


#### Create Multiple List


```python
lst_1 = [1, 2, 3, 4]
lst_2 = [2, 4, 6, 8]
lst_3 = [3, 6, 9, 12]
```

#### Map Add Function to Multiple Lists


```python
add_elements = map(lambda x, y ,z : x + y + z, lst_1, lst_2, lst_3)
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

