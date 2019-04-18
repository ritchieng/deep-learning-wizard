
# C++

!!! tip "Run Jupyter Notebook"
    You can run the code for this section in this [jupyter notebook link](https://github.com/ritchieng/deep-learning-wizard/blob/master/docs/programming/cpp/cpp.ipynb).

## Installation of Interactive C++17

Xeus-Cling is a game-changer where similar to Python Jupyter Notebooks, we can run C++ Jupyter Notebooks now. Run the following bash commands in sequence to create a C++ kernel for Jupyter Notebook.

```bash
conda create -n cpp
source activate cpp
conda install -c conda-forge xeus-cling
jupyter kernelspec install --user /home/ritchie/miniconda3/envs/cpp/share/jupyter/kernels/xcpp17
jupyter notebook
```

## Printing

### Printing Single Line


```c++
// When you compile, the preprocessor runs and acts on all line with the pound key first

// This is a preprocessor instruction that
// essentially places the file with the name iostream into this spot
#include <iostream>
```


```c++
// Print one line
std::cout << "Using Xeus Cling" << std::endl;
```

    Using Xeus Cling


### Printing 2 Lines


```c++
// Print two lines
std::cout << "First Line \nSecond Line" << std::endl;
```

    First Line 
    Second Line


### Printing with Tabs


```c++
// Print numbers with nicely formatted tabs
std::cout << "One hundred:\t";
std::cout << (float) 1000/10 << std::endl;

std::cout << "Two hundred:\t";
std::cout << (double) 2000/10 << std::endl;

std::cout << "Three hundred:\t";
std::cout << (double) 3000/10 << std::endl;
```

    One hundred:	100
    Two hundred:	200
    Three hundred:	300


### Easier Printing (subjective) with Namespaces
- Gets irritating to use `std` in front of `cout` and `endl` to keep printing so we can use namespaces


```c++
using namespace std;
cout << "No need for messy std::" << endl;
```

    No need for messy std::


## Variables

### General memory management
In C++, you always need to determine each variable's type so the compiler will know how much memory (RAM) to allocate to the variable

### Types of Variables
| Type  |  Bytes (Size) | Range of Values|
|-------|--------------|----------------|
| char | 1 | 0 to 255 or -127 to 127|
| unsigned char | 1 | -127 to 127 |
| signed char | 1 | 0 to 255 |
| bool | 1 | True or False |
| int | 4 | -2,147,483,648 to 2,147,483,647 |
| unsigned int | 4 | 0 to 4,294,967,295 |
| signed int | 4 | -2,147,483,648 to 2,147,483,647 |
| short int | 2 | -32,768 to 32,767 |
| long int | 4 | -2,147,483,648 to 2,147,483,647 |
| float | 4 | 3.4e-38 to 3.4e38 |
| double | 8 | 1.7e-308 to 1.7e-08|
| long double | 8 | 1.7e-308 to 1.7e-08|

### Integer sizes
- Typical sizes
    - Short integer: 2 bytes (will be smaller than long)
        - Limitation is that the value of this short integer has a max value, you should be cautious of using short integers
    - Long integer: 4 bytes
    - Integer: 2 or 4 bytes
- Notes
    - Technically these sizes can vary depending on your processor (32/64 bit) and compiler
    - You should not assume the sizes but the hierarchy of sizes will not change (short memory < long memory)

#### Integer size


```c++
cout << "Size of an integer: " << sizeof (int);
```

    Size of an integer: 4

#### Short integer size


```c++
cout << "Size of a short integer: " << sizeof (short int);
```

    Size of a short integer: 2

#### Long integer size


```c++
cout << "Size of an long integer: " << sizeof (long int);
```

    Size of an long integer: 8

### Unsigned or signed integers
- Unsigned integers: only can hold positive integers
- Signed integers: can hold positive/negative integers

#### Signed short integer


```c++
cout << "Size of an signed short integer: " << sizeof (signed short int);
```

    Size of an signed short integer: 2

#### Unsigned short integer


```c++
cout << "Size of an unsigned short integer: " << sizeof (unsigned short int);
```

    Size of an unsigned short integer: 2

#### Signed long integer


```c++
cout << "Size of an signed long integer: " << sizeof (signed long int);
```

    Size of an signed long integer: 8

#### Unsigned long integer


```c++
cout << "Size of an unsigned long integer: " << sizeof (unsigned long int);
```

    Size of an unsigned long integer: 8

## Functions

### Function Without Return Value


```c++
// Usage of void when the function does not return anything
// In this exmaple, this function prints out the multiplication result of two given numbers
void MultiplyTwoNumbers(int firstNum, int secondNum)
{
    // Define variable as integer type
    long int value;
    value = firstNum * secondNum;
    std::cout << value << std::endl; 
}
```

#### Multiply Two Numbers 3 and 2


```c++
MultiplyTwoNumbers(3, 2)
```

    6


#### Multiply Two Numbers 6 and 2


```c++
MultiplyTwoNumbers(6, 2)
```

    12


### Aliases Function

Say we want the variable `value` to be of type `unsigned short int` such that it's 2 bytes and can hold 2x the range of values compared to just `short int`. We can use `typedef` as an alias.

| Type  |  Bytes (Size) | Range of Values|
|-------|--------------|----------------|
| short int | 2 | -32,768 to 32,767 |
| unsigned short int | 2 |  0 to 65,536 |


```c++
// Usage of void when the function does not return anything
// In this exmaple, this function prints out the multiplication result of two given numbers
void MultiplyTwoNumbersWithAlias(int firstNum, int secondNum)
{
    // Using an alias
    typedef unsigned short int ushortint;
    // initializing value variable with ushortint type
    ushortint value;
    value = firstNum * secondNum;
    std::cout << value << std::endl; 
}
```

#### Multiply Two Numbers 10 and 10


```c++
MultiplyTwoNumbersWithAlias(10, 10)
```

    100


#### Multiply Two Numbers 1000 and 65


```c++
MultiplyTwoNumbersWithAlias(1000, 65)
```

    65000


#### Multiply Two Numbers 1000 and 67
- Notice how you don't get 67,000? This is because our variable `value` of `ushortint` type can only hold values up to the integer 65,536.
- What this returns is the remainder of 67,000 - 65,536 = 1464


```c++
MultiplyTwoNumbersWithAlias(1000, 67)
```

    1464



```c++
std::cout << 67 * 1000 - 65536 << std::endl;
```

    1464


###  Function with Return Value
Unlike functions without return values where we use `void` to declare the function, here we use `int` to declare our function that returns values.


```c++
// In this exmaple, this function returns the value of the multiplication of two numbers
int MultiplyTwoNumbersNoPrint(int firstNum, int secondNum)
{
    // Define variable as integer type
    long int value;
    value = firstNum * secondNum;
    return value;
}
```

#### Call Function


```c++
// Declare variable with type
long int returnValue;
// Call function
returnValue = MultiplyTwoNumbersNoPrint(10, 2);
// Print variable
std::cout << returnValue << std::endl;
```

    20

