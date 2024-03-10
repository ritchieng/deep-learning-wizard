---
comments: true
---

# Calculus wth Sympy

In this section, we will be covering basic calculus theory and using sympy to solve the related equations.

## Calculus: Differentiation (Theory)

### Basic Rules

---
$$
f(x) = x^n
\\ f\prime(x) = nx^{x-1}
$$

---
$$
f(x) = c
\\ f\prime(x) = 0
$$

---
$$
f(x) = \cos(x)
\\ f\prime(x) = -\sin(x)
$$

---
$$
f(x) = \cosh(x)
\\ f\prime(x) = \sinh(x)
$$

---
$$
f(x) = \sin(x)
\\ f\prime(x) = \cos(x)
$$

---
$$
f(x) = \sinh(x)
\\ f\prime(x) = \cosh(x)
$$

---
$$
f(x) = \ln (x)
\\ f\prime(x) = \frac{1}{x}
$$

### Sum Difference Rule

Given

$$ f(x) = g(x) \pm h(x) $$

Then

$$ f\prime(x) = g\prime(x) \pm h\prime(x) $$

### Product Rule

Given a function with two parts

$$ f(x) = g(x) \cdot h(x) $$

First derivative

$$ f\prime(x) = g\prime(x) \cdot h(x) + g(x) \cdot h\prime(x) $$

Given three parts, we always use the product rule to keep it to 2 parts

$$ f(x) = g(x) \cdot h(x) \cdot i(x) $$

$$ f'(x) = \big[g(x) \cdot h(x)\big]\prime \cdot i(x) + \big[g(x) \cdot h(x) \big] \cdot i\prime(x) $$

### Quotient Rule

Given

$$ f(x) = \frac{g(x)}{h(x)} $$

Then

$$ f\prime(x) = \frac{g\prime(x) \cdot h(x) - g(x) \cdot h\prime(x)}{h^2(x)} $$

### Logarithmic Rule

Given

$$f(x) = ln(g(x))$$

Then

$$ f\prime(x) = \frac{g\prime(x)}{g(x)} $$

### Exponential Rule

Given

$$ f(x) = \exp \big(g(x)\big) $$

Then

$$ f\prime(x) = g\prime(x) \exp \big(g(x)\big) $$

### Partial Derivative

Given

$$ f(x) = g(x_{i}, x_{i+1}, \dots, x_n) $$

When computing the partial derivative of $x_i$ with respect to (w.r.t.) the multivariable function $f(x_i \dots x_n)$, treat the rest of the variables as constant.

## Calculus: Integration (Theory)

### Basic Rules

Given function $f(x)$ and integral of that function $F(x)$, then basic rules include

---
$$
f(x) = x^n
\\ F(x) = \frac{x^{n+1}}{n+1} + c
$$

---
$$
f(x) = 0
\\ F(x) = c
$$

---
$$
f(x) = e^x
\\ F(x) = e^x + c
$$

---
$$
f(x) = a^x
\\ F(x) = \frac{a^x}{\ln a} + c
$$

---
$$
f(x) = \sin(x)
\\ F(x) = -\cos(x) + c
$$

---
$$
f(x) = \sinh(x)
\\ F(x) = \cosh(x) + c
$$

---
$$
f(x) = \cos(x)
\\ F(x) = \sin(x) + c
$$

---
$$
f(x) = \cosh(x)
\\ F(x) = \sinh(x) + c
$$

---
$$
f(x) = \frac{1}{x}
\\ F(x) = \ln\left|x\right| + c
$$

---
$$
f(x) = m
\\ F(x) = mx + c
$$

---
$$
f(x) = \frac{1}{x^2 + a^2}
\\ F(x) = (\frac{1}{a}) \cdot \arctan(\frac{x}{a}) + c
$$

---
$$
f(x) = \frac{1}{\sqrt{a^2 - x^2}}
\\ F(x) = \arcsin(\frac{x}{a}) + c
$$

### Sum Difference Rule

Given

$$ f(x) = g(x) \pm h(x) $$

Then

$$ F(x) = \int \big( g(x) \pm h(x) \big) \, dx$$

### Integration by Parts

$$
F(x) = \int u\ v\prime \ dx = u\ v - \int v\ u\prime \ dx
$$

### Definite Integral Rules

#### Zero integral when not moving integration point

$$
\int _a ^a f(x) \ dx = 0
$$

#### Definite integral's interval switching becomes negative

$$
\int _a ^b f(x) \ dx = - \int _b ^a f(x) \ dx 
$$

#### Definite integral decomposed into parts

$$
\int _a ^b f(x) \ dx = \int _a ^c f(x) \ dx - \int _b ^c f(x) \ dx
$$

#### Substitution method for solving definite integrals

---

##### Substitution Example 1

Given
$$ f(x) = x \cos(x^2 + 1) $$

Then definite integral from -1 to 1 is 
$$ F(x)_{-1}^1 = \int _{-1} ^ 1 x \cos(x^2 + 1) \ dx $$

Modifying the equation to make $du$ and $u$ substitution method results in
$$ F(x)_{-1}^1 = \frac{1}{2} \int _{-1} ^ 1 2x \cos(x^2 + 1) \ dx $$

Let $u$ be
$$ u = x^2 +1  $$

Let $du$ be
$$ du = 2x $$

New limits given -1 and 1
$$ u = (-1)^2 + 1 = 2 $$
$$ u = (1)^2 + 1 = 2 $$

Then by zero integral rule
$$ F(x)_{-1}^1 = \int _2 ^2 cos(u) du = 0 $$

---

##### Substitution Example 2

Given
$$ h(x) = x(x+3)^{\frac{1}{2}} $$

Then definite integral from -1 to 1 is
$$ H(x)_{-1}^1 = \int _{-1} ^ 1 x(x+3)^{\frac{1}{2}} $$

Let $u$ be
$$ u = x + 3 $$
$$ x = u - 3 $$

Let $du$ be
$$ du = 1 $$

New limits given -1 and 1
$$ u = -1 + 3 = 2 $$
$$ u = 1 + 3 = 4 $$

Then

$$ H(x)_{-1}^1 = \int _{2} ^ 4 (u-3)(u)^{\frac{1}{2}} \ du $$

$$ H(x)_{-1}^1 = \int _{2} ^ 4 u^{\frac{3}{2}} - 3u^{\frac{1}{2}} \ du $$

$$ H(x)_{-1}^1 = \big[ \frac{u^{\frac{5}{2}}}{\frac{5}{2}} - \frac{3u^{\frac{3}{2}}}{\frac{3}{2}} \big]_2 ^4 $$

$$ H(x)_{-1}^1 = \frac{2}{5} [-8 + 6\sqrt(2)] $$

---

##### Substitution Example 3

!!! note "Keep on substituting!"
    You can keep running the substitution rule multiple times, where you can go from $u$ to $z$ to etc. to solve the definite integral. It is important to take note that your limits change every time you run the substitution rule once!


### Multiple Integrals

This is used for functions with multiple variables. Hence, instead of the usual single variable where the definite integral represents the area under the curve, multiple integrals calculate hypervolumes with multiple dimensional functions.

The key is in integrating step by step, when integrating with respect to each variable, the rest of the variables act as constants.

#### Double Integral
$$V = \int _{x=a}^{x=b} \int _{y=c}^{y=d} f(x,y) \ dy \ dx$$

#### Triple Integral

$$V = \int _{x=a}^{x=b} \int _{y=c}^{y=d} \int _{z=e}^{z=f} f(x,y,z) \ dz \ dy \ dx$$

## Calculus: Taylor Expansion (Theory)

### Taylor Expansion: Single Variable

Given a function that has continuous derivatives up to $(n+1)$ order, the function can be expanded with the following equation:

$$ F(x) = \sum^{\infty}_{n=0} \frac{F^{(n)}(a)}{n!}\ (x-a)^n $$

$$ F(x) = F(a) + F\prime (x)(x-a) + \frac{F\prime\prime(a)}{2!}(x-a)^2 + \cdots + \frac{F(n)(a)}{n!} + R_n(x) $$

If not expanded till $\infty$ then there will be a remainder of $R_n(x)$

### Taylor Expansion: Multiple Variables

In the case of 2 variables $(x, y)$, we can expand the multivariate equation with the generalized Taylor expansion equation:

$$
F(x, y) = \\
F(a, b) + \big[ \frac{\partial}{\partial x} F(x, y)_{a,b} (x-a) + \frac{\partial}{\partial y} F(x, y)_{a,b} ) (y - b)  \big] \\
+ \frac{1}{2!} \big[ \frac{\partial ^2}{\partial x^2} F(x, y)_{a,b} (x-a)^2 + \frac{\partial ^2}{\partial y^2} F(x, y)_{a,b}(y-b)^2 \big] \\
+ \frac{\partial ^2}{\partial x \partial y} F(x, y)_{a,b} (x-a)(y-b) + R_2 (x,y)
$$

### Maclaurin Expansion

It is simply a Taylor expansion about the point 0.