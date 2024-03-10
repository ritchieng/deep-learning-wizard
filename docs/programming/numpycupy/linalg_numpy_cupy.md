---
comments: true
---

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
    - add/subtract every individual element $a_{ij}$ of the first matrix $A_{M \times N}$ with $b_{ij}$ of the second matrix $B_{M \times N}$
- Scalar Multiplication with Matrix
    - multiply scalar $\lambda$ with every individual element $a_{ij}$ of the matrix $A_{M \times N}$
- Matrix Multiplication
    - can only be done where $A_{M \times N}$ and $B_{N \times O}$ have a coinciding dimension $N$ which is the same to yield a new matrix $C_{N \times O}$ of $N \times O$ dimension
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

## Linear System of Equations in Matrices

We can express many linear equations in the form of matrices for example $AX= B$

Where we have $A$ representing our parameters, $B$ representing our input variables and $B$ representing our constant/bias variables.

$$
A = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1N} \\ a_{21} & a_{22} & \cdots & a_{2N} \\ \vdots & \vdots & \ddots & \vdots \\ a_{M1} & a_{M2} & \cdots & a_{MN} \end{bmatrix}, X = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_M \end{bmatrix}, B = \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_M \end{bmatrix}
$$

If we find $\det (A)$ to be non-zero where the matrix is a non-singular matrix, the system of equations would have a unique solution. Hence, given $A$ and $B$, we can find out the unique solution in $B$.

$$ 
AX = B \\
A^{-1}AX = A^{-1}B \\
IX = A^{-1}B \\
X = A^{-1}B
$$

### Solving System of Equations for Square and Non-Singular

If the matrix A is square and $det(A) \neq 0$ (non-singular, inverse matrix can be calculated), then Cramer's rule can be applied to solve the linear system of equations.

#### Process for a 2 x 2 Non-Singular Matrix A
- Replace First Column of $A$ with $B$ to get $A_1$
$$ x_1 = \frac{\det(A_1)}{\det(A)} $$
- Replace Second Column of $A$ with $B$ to get $A_2$
$$ x_2 = \frac{\det(A_2)}{\det(A)} $$
- Repeat till all columns covered for bigger matrices

## Eigenvalue (characteristic root) and eigenvalue (characteristic value)

Given a square matrix $A$, eigenvalue $\lambda$ and eigenvector $v$ where $v \neq 0$, we have:

$$
A v = \lambda v \\
A v = \lambda I v \\
Av - \lambda I v = 0 \\
(A - \lambda I)v = 0 \\
$$

Since  $v \neq 0$ then we have characteristic matrix $(A - \lambda I) = 0$.

### Solving eigens
- We can solve for the determinant of the characteristic matrix (characteristic polynomial) $|A - \lambda I| = 0$ through this characteristic equation.
    - We will get multiple values of $\lambda$
- Substitute $\lambda$ into $(A - \lambda I)v = 0$, solve for $v$
    - If infinite solution (no constant values for x and y), impose uniqueness with $v\prime v  = 1$ in this case $\begin{bmatrix} x & y \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = x^2 + y^2 = 1$
    - Substitute y into $v\prime v  = 1$ to solve for x, then solve for y
    - Substitute x and y to solve for $v$
- Simple eigenvector $\lambda _1 \neq \lambda _2$ 
- Repeated/double eigenvector $\lambda _1 = \lambda _2$

### Properties of eigenvalue and eigenvector
- If $|A - \lambda I| = 0$, singular therefore infinite solutions, hence:
    - $\lambda \gt 0$: positive definite
    - $\lambda \ge 0$: positive semi-definite
    - $\lambda \lt 0$: negative definite
    - $\lambda \le 0$: negative semi-definite
