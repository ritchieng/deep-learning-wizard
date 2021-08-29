# Linear Algebra with NumPy and CuPy

In this section, we will be covering linear algebra and using numpy for CPU-based matrix manipulation and cupy for GPU-based matrix manipulation.

## Types of Matrices

- M x N Matrix
    - M rows and N columns
- 1 x N Matrix (Row Vector)
    - 1 row and N columns
- M x 1 Matrix (Column Vector)
    - M rows and 1 column
- 1 x 1 Matrix (Scalar)
    - 1 row and 1 column
- N x M Matrix from M x N Matrix (Transposed Matrix)
    - Swapping M rows to columns and N columns to rows where matrix $A$ then becomes transposed matrix $A\prime$
- Symmetric Matrix
    - M x N Matrix is equal to N x M Transposed Matrix
- M x M Matrix (Square Matrix)
    - M rows = N columns
- Diagonal Matrix
    - all matrix elements are zeroes except for the diagonal
- Identity matrix
    - all matrix elements are zeroes except for the diagonals being 1's 

## Multiplication and Subtraction

- Element-wise Matrix Addition/Subtraction
    - add/subtract every individual element $a_{ij}$ of the first matrix $A_{M \times N}$ with $b_{ij}$ of the second matrix $B_{M \times N}Scalar Multiplication with Matrix: multiply scalar $\lambda$ with every individual element $a_{ij}$ of the matrix $A_{M \times N}$
- Matrix Multiplication
    - can only be done where $A_{M \times N}$ and $B_{N \times O}$ have a coinciding dimension $N$ which is the same to yield a new matrix $C_{N \times O}$ of $N \times O $ dimension
    - new element $c_{ij} = \sum _{q=1} ^N a_{iq} b_{qj}$
    - essentially the first row and first column element of the new matrix is equal to the summation element-wise multiplication of the first row of matrix $A$ with the first column of matrix $B$

## Determinant of Square Matrix

### Determinant of 2 x 2 Square Matrix
The determinant of a 2x2 square matrix can be derived by multiplying all the elements on the main diagonal and subtracting the multiplication of all the elements in the other diagonal which will result in a scalar value.

$$ A = \begin{bmatrix}a & b\\c & d\end{bmatrix} $$

$$\det(A) = | A | =  ad - bc $$

### Determinant of 3 x 3 Square Matrix

$$ \begin{bmatrix} a & b &c \\ d& e &f \\ g& h &i \end{bmatrix} \\
= a (-1)^{(1+1)}\det \begin{bmatrix} e & f\\ h & i \end{bmatrix} + b (-1)^{(1+2)} \det \begin{bmatrix} d & f\\ g & i \end{bmatrix} + c (-1)^{(1+3)} \det \begin{bmatrix} d & e\\ g & h \end{bmatrix} $$

### Determinant Types
* singular matrix: $\det(A) = 0$ where at least 2 rows/columns are linearly dependent
* non-singular: $\det(A) \neq 0$ where no rows/columns are linearly dependent (matrix has full rank)


## Inverse of a Non-Singular Square Matrix

### Unique Property

The inverse can be calculated only if it is a non-singular square matrix hence it is useful to first calculate the determinant. Also the inverse of the matrix yields a useful and unique property where

$$ AA^{-1} = A^{-1}A = I $$

### Inverse and Identity of 2 x 2 Non-Singular Square Matrix

$$ A=\begin{bmatrix}a & b \\c & d \end{bmatrix} $$ 

$$ A^{-1}=\frac{1}{ad-bc}\begin{bmatrix}d & -b \\-c & a \end{bmatrix} $$

$$ \begin{array}{lcl}AA^{-1}&=&\begin{bmatrix}a & b \\c & d \end{bmatrix}\frac{1}{ad-bc}\begin{bmatrix}d & -b \\-c & a \end{bmatrix}\\ &=&\frac{1}{ad-bc}\begin{bmatrix}ad+b(-c) & a(-b)+b(a) \\cd+d(-c) & c(-b)+d(a) \end{bmatrix}\\ &=&\begin{bmatrix}1 & 0\\0 & 1 \end{bmatrix}\end{array} $$
