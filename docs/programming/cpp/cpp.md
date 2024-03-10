---
comments: true
---

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

## Constants

### Literal Constants


```c++
int varOne = 20;
std::cout << varOne << std::endl;
```

    20


### Enumerated Constants
This enables you to create a new type! In this example we create a new type `directions` containing `Up`, `Down`, `Left`, and `Right`.


```c++
enum directions {Up, Down, Left, Right};
    
directions goWhere;
goWhere = Right;

if (goWhere == Right)
    std::cout << "Go right" << std::endl;
```

    Go right


## Functions

### Function Without Return Value
Syntax generally follows `void FunctionName(argOne, argTwo)` to define the function followed by `FuntionName()` to call the function.


```c++
// Usage of void when the function does not return anything
// In this example, this function prints out the multiplication result of two given numbers
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


###  Function Inner Workings
- Essentially our lines of codes translates to instruction pointers with unique memory addresses.
- Execution of instruction pointers operates on a "LIFO" basis, last in first out.
    - Oversimplifying here, in our example, the last line is taken off first and it follows up


```c++
// Code Space
int varOneTest = 10; // Instruction pointer 100
std::cout << varOneTest << std::endl; // Instruction Pointer 102
```

    10





    @0x7fa6b7de5460



## Arrays

An array contains a sequence of elements with the same data type. 

### Creating an Array


```c++
// This is how you declare an array of 50 elements each of type double
double DoubleArray[50];
```

#### Accessing Array's Elements

##### First Element


```c++
// This access the first array 
std::cout << DoubleArray[0] << std::endl;
```

    0


##### Last Element


```c++
std::cout << DoubleArray[49] << std::endl;
```

    0


##### First 10 Elements


```c++
// In steps of 1
for (int i=0; i<10; i++)
{
    // This is how you print a mix of characters and declared variables
    std::cout << "Element " << i << " contains " <<  DoubleArray[i] << std::endl;
}
```

    Element 0 contains 0
    Element 1 contains 0
    Element 2 contains 0
    Element 3 contains 0
    Element 4 contains 0
    Element 5 contains 0
    Element 6 contains 0
    Element 7 contains 0
    Element 8 contains 0
    Element 9 contains 0



```c++
// In steps of 2
for (int i=0; i<10; i+=2)
{
    // This is how you print a mix of characters and declared variables
    std::cout << "Element " << i << " contains " <<  DoubleArray[i] << std::endl;
}
```

    Element 0 contains 0
    Element 2 contains 0
    Element 4 contains 0
    Element 6 contains 0
    Element 8 contains 0


##### Going Beyond The Array's Length
This will return a warning that it's past the end of the array

```c++
std::cout << DoubleArray[50] << std::endl;
```
```bash
input_line_36:2:15: warning: array index 50 is past the end of the array (which contains 50 elements)
      [-Warray-bounds]
 std::cout << DoubleArray[50] << std::endl;
              ^           ~~
input_line_32:3:1: note: array 'DoubleArray' declared here
double DoubleArray[50];
^
4.94066e-323
```

### Arrays with Enumeration


```c++
enum directionsNew {up, down, left, right, individualDirections};
    
int directionsArray[individualDirections] = {1, 2, 3, 4};

std::cout << "Up value:\t" << directionsArray[up];
std::cout << "\nDown value:\t" << directionsArray[down];
std::cout << "\nLeft value:\t" << directionsArray[left];
std::cout << "\nRight value:\t" << directionsArray[right];
// This is the number of elements in the array
std::cout << "\nNum value:\t" << sizeof(directionsArray) / sizeof(directionsArray[0]) << std::endl;
```

    Up value:	1
    Down value:	2
    Left value:	3
    Right value:	4
    Num value:	4


### Arrays with >1 Dimension (Tensors)

#### Multi Dimension Array with Numbers


```c++
// This is how you declare a multi-dimensional array of 5x5 elements each of type double
double multiDimArray[5][5] = {
    {1, 2, 3, 4, 5},
    {2, 2, 3, 4, 5},
    {3, 2, 3, 4, 5},
    {4, 2, 3, 4, 5},
    {5, 2, 3, 4, 5}
};

// Print each row of our 5x5 multi-dimensional array
for (int i=0; i<5; i++)
{
    for (int j=0; j<5; j++)
    {
        std::cout << multiDimArray[i][j];
    };
    std::cout << "\n" << std::endl;
};
```

    12345
    
    22345
    
    32345
    
    42345
    
    52345
    


#### Multi Dimension Array with Characters


```c++
// This is how you declare a multi-dimensional array of 5x5 elements each of type char
char multiDimArrayChars[5][5] = {
    {'a', 'b', 'c', 'd', 'e'},
    {'b', 'b', 'c', 'd', 'e'},
    {'c', 'b', 'c', 'd', 'e'},
    {'d', 'b', 'c', 'd', 'e'},
    {'e', 'b', 'c', 'd', 'e'},
};

// Print each row of our 5x5 multi-dimensional array
for (int i=0; i<5; i++)
{
    for (int j=0; j<5; j++)
    {
        std::cout << multiDimArrayChars[i][j];
    };
    std::cout << "\n" << std::endl;
};
```

    abcde
    
    bbcde
    
    cbcde
    
    dbcde
    
    ebcde
    


### Copy Arrays

#### Copy Number Arrays


```c++
double ArrayNumOne[] = {1, 2, 3, 4, 5};
double ArrayNumTwo[5];

// Use namespace std so it's cleaner
using namespace std;

// Copy array with copy()
copy(begin(ArrayNumOne), end(ArrayNumOne), begin(ArrayNumTwo));

// Print double type array with copy()
copy(begin(ArrayNumTwo), end(ArrayNumTwo), ostream_iterator<double>(cout, "\n"));
```

    1
    2
    3
    4
    5


#### Copy String Arrays


```c++
char ArrayCharOne[] = {'a', 'b', 'c', 'd', 'e'};
char ArrayCharTwo[5];

// Use namespace std so it's cleaner
using namespace std;

// Copy array with copy()
copy(begin(ArrayCharOne), end(ArrayCharOne), begin(ArrayCharTwo));

// Print char type array with copy()
copy(begin(ArrayCharTwo), end(ArrayCharTwo), ostream_iterator<char>(cout, "\n"));
```

    a
    b
    c
    d
    e


## Mathematical Operators
- Add, subtract, multiply, divide and modulus

### Add


```c++
double addStatement = 5 + 5;
std::cout << addStatement;
```

    10

### Subtract


```c++
double subtractStatement = 5 - 5;
std::cout << subtractStatement;
```

    0

### Divide


```c++
double divideStatement = 5 / 5;
std::cout << divideStatement;
```

    1

### Multiply


```c++
double multiplyStatement = 5 * 5;
std::cout << multiplyStatement;
```

    25

### Modulus
- Gets the remainder of the division


```c++
double modulusStatement = 8 % 5;
std::cout << modulusStatement;
```

    3

### Exponent
- Base to the power of something, this requires a new package called `<cmath>` that we want to include.


```c++
#include <cmath>

void SquareNumber(int baseNum, int exponentNum)
{
    // Square the locally scoped variable with 2
    int squaredNumber;
    squaredNumber = pow(baseNum, exponentNum);
    std::cout << "Base of 2 with exponent of 2 gives: " << squaredNumber << std::endl;
}

SquareNumber(2, 2)
```

    Base of 2 with exponent of 2 gives: 4


### Incrementing/Decrementing
- 3 ways to do this, from least to most verbose

#### Methods

##### Method 1


```c++
double idx = 1;
idx++;
std::cout << idx;
```

    2

##### Method 2


```c++
idx += 1;
std::cout << idx;
```

    3

##### Method 3


```c++
idx = idx + 1;
std::cout << idx;
```

    4

#### Prefix/Postfix


##### Prefix
- This will change both incremented variable and the new variable you assign the incremented variable to
- Summary: both variables will have the same values


```c++
// Instantiate
double a = 1;
double b = 1;

// Print original values
cout << "Old a:\t" << a << "\n";
cout << "Old b:\t" << b << "\n";

// Prefix increment
a = ++b;

// Print new values
cout << "New a:\t" << a << "\n";
cout << "New b:\t" << b << "\n";
```

    Old a:	1
    Old b:	1
    New a:	2
    New b:	2


##### Postfix
- This will change only the incremented variable but not the variable it's assigned to
- Summary: incremented variable will change but not the variable it was assigned to


```c++
// Instantiate
double c = 2;
double d = 2;

// Print original values
cout << "Old c:\t" << c << "\n";
cout << "Old d:\t" << d << "\n";

// Prefix increment
c = d--;

// Print new values, notice how only d decremented? c which is what d is assigned to doesn't change.
cout << "New c:\t" << c << "\n";
cout << "New d:\t" << d << "\n";
```

    Old c:	2
    Old d:	2
    New c:	2
    New d:	1


## Conditional Statements

#### If


```c++
int maxValue = 10;
    
// Increment till 10
for (int i=0; i<=10; i+=2)
{
    // Stop if the number reaches 10 (inclusive)
    if (i == maxValue)
    {
        cout << "Reached max value!";
        cout << "\nValue is " << i << endl;
    };
};
```

    Reached max value!
    Value is 10


#### Else


```c++
int newMaxValue = 20;
    
// Increment till 10
for (int i=0; i<=10; i+=2)
{
    // Stop if the number reaches 10 (inclusive)
    if (i == newMaxValue)
    {
        cout << "Reached max value!";
        cout << "\nValue is " << i << endl;
    }
    // Else print current value
    else
    {
        cout << "\nCurrent Value is " << i << endl;
    }
}
```

    
    Current Value is 0
    
    Current Value is 2
    
    Current Value is 4
    
    Current Value is 6
    
    Current Value is 8
    
    Current Value is 10


## Logical Operators

#### And


```c++
int varOneNew = 10;
int varTwo = 10;
int varCheckOne = 10;
int varCheckTwo = 5;

// This should print out
if ((varOneNew == varCheckOne) && (varTwo == varCheckOne))
{
    std::cout << "Both values equal to 10!" << std::endl;
}

// This should not print out as varTwo does not equal to 5
if ((varOneNew == varCheckOne) && (varTwo == varCheckTwo))
{
    std::cout << "VarOneNew equals to 10, VarTwo equals to 5" << std::endl;
}
```

    Both values equal to 10!


#### Or


```c++
// On the contrary, this exact same statement would print out
// as VarOne is equal to 10 and we are using an OR operator
if ((varOneNew == varCheckOne) || (varTwo == varCheckTwo))
{
    std::cout << "VarOneNew equals to 10 or VarTwo equals to 5" << std::endl;
}
```

    VarOneNew equals to 10 or VarTwo equals to 5


#### Not


```c++
// This would print out as VarTwo is not equal to 5
if (varTwo != varCheckTwo)
{
    std::cout << "VarTwo (10) is not equal to VarCheckTwo (5)." << std::endl;
}
```

    VarTwo (10) is not equal to VarCheckTwo (5).


## Getting User Input


```c++
using namespace std;
long double inputOne, inputTwo;
cout << "This program multiplies 2 given numbers\n";
cout << "Enter first number: \n";
cin >> inputOne;
cout << "Enter second number: \n";
cin >> inputTwo;
cout << "Multiplication value: " << inputOne * inputTwo << endl;
```

    This program multiplies 2 given numbers
    Enter first number: 
    10
    Enter second number: 
    10
    Multiplication value: 100


## Loops

### For Loop


```c++
for (int i=0; i<10; i+=1)
{
    cout << "Value of i is: " << i << endl;
}
```

    Value of i is: 0
    Value of i is: 1
    Value of i is: 2
    Value of i is: 3
    Value of i is: 4
    Value of i is: 5
    Value of i is: 6
    Value of i is: 7
    Value of i is: 8
    Value of i is: 9


### While Loop


```c++
int idxWhile = 0;

while (idxWhile < 10)
{
    idxWhile += 1;
    cout << "Value of while loop i is: " << idxWhile << endl;
}
```

    Value of while loop i is: 1
    Value of while loop i is: 2
    Value of while loop i is: 3
    Value of while loop i is: 4
    Value of while loop i is: 5
    Value of while loop i is: 6
    Value of while loop i is: 7
    Value of while loop i is: 8
    Value of while loop i is: 9
    Value of while loop i is: 10


### While Loop with Continue/Break



```c++
int idxWhileNew = 0;

while (idxWhileNew < 100)
{
    idxWhileNew += 1;
    cout << "Value of while loop i is: " << idxWhile << endl;
    
    if (idxWhileNew == 10)
    {
        cout << "Max value of 10 reached!" << endl;
        break;
    }
}
```

    Value of while loop i is: 10
    Value of while loop i is: 10
    Value of while loop i is: 10
    Value of while loop i is: 10
    Value of while loop i is: 10
    Value of while loop i is: 10
    Value of while loop i is: 10
    Value of while loop i is: 10
    Value of while loop i is: 10
    Value of while loop i is: 10
    Max value of 10 reached!

