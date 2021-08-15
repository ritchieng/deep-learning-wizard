# Math for AI

## Calculus: Differentiation

### Basic Rules

$$

f(x) = x^n
\\ f\prime(x) = nx^{x-1}

\\ \dots

\\ f(x) = c
\\ f\prime(x) = 0

\\ \dots

\\ f(x) = \cos(x)
\\ f\prime(x) = -\sin(x)

\\ \dots

\\ f(x) = \cosh(x)
\\ f\prime(x) = \sinh(x)

\\ \dots

\\ f(x) = \sin(x)
\\ f\prime(x) = \cos(x)

\\  \dots

\\ f(x) = \sinh(x)
\\ f\prime(x) = \cosh(x)

\\ \dots

\\ f(x) = \ln (x)
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

$$f(x) = \frac{g(x)}{h(x)}$$

Then

$$f\prime(x) = \frac{g\prime(x) \cdot h(x) - g(x) \cdot h\prime(x)}{h^2(x)}$$

### Logarithmic Rule

Given

$$f(x) = ln(g(x))$$

Then

$$f\prime(x) = \frac{g\prime(x)}{g(x)}$$

### Exponential Rule

Given
$$f(x) = \exp \big(g(x)\big)$$

Then
$$f\prime(x) = g\prime(x) \exp \big(g(x)\big)$$

### Partial Derivative

Given

$$ f(x) = g(x_{i}, x_{i+1}, \dots, x_n) $$

When computing the partial derivative of $x_i$ with respect to (w.r.t.) the multivariable function $f(x_i \dots x_n)$, treat the rest of the variables as constant.

## Calculus: Integration

### Basic Rules

Given function $f(x)$ and integral of that function $F(x)$, then basic rules include

$$

f(x) = x^n
\\ F(x) = \frac{x^{n+1}}{n+1} + c

\\ \dots

\\ f(x) = 0
\\ F(x) = c

\\ \dots

\\ f(x) = \sin(x)
\\ F(x) = -\cos(x) + c

\\ \dots

\\ f(x) = \sinh(x)
\\ F(x) = \cosh(x) + c

\\ \dots

\\ f(x) = \cos(x)
\\ F(x) = \sin(x) + c

\\ \dots

\\ f(x) = \cosh(x)
\\ F(x) = \sinh(x) + c

\\ \dots

\\ f(x) = \frac{1}{x}
\\ F(x) = \ln(x) + c

\\ \dots

\\ f(x) = m
\\ F(x) = mx + c

\\ \dots

\\ f(x) = \frac{1}{x^2 + a^2}
\\ F(x) = (\frac{1}{a}) \cdot \arctan(\frac{x}{a}) + c

\\ \dots

\\ f(x) = \frac{1}{\sqrt{a^2 - x^2}}
\\ F(x) = \arcsin(\frac{x}{a}) + c

$$

### Sum Difference Rule

Given

$$ f(x) = g(x) \pm h(x) $$

Then

$$ F(x) = \int \big( g(x) \pm h(x) \big) \, dx$$

