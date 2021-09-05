# Asset Management Theory

## FV, PV, NPV and IRR

### Future Value (FV) and Present Value (PV)

Given the present value, rate of return and time, we can calulate the future value (FV) with $FV = PV (1+r)^t$

### Net Present Value (NPV)
- It is the sum of discounted cashflow given by $NPV = \sum_{t=0} ^ T \frac{CF_t}{(1+r)^t}$
  - $CF_t$ is the difference between the benefits and costs the project incurs each year
  - Such a metric is often used to determine if it's worth embarking on the project
  - Dependent on market rates, and is often used to compare different projects

### Internal Rate of Return (IRR)
- To calculate it, just set $NPV=0$ and solve for $r$
  - It can be a tedious process, so just use available solvers in Python
- Simply calculating the minimum rate of return to cover the project's costs
- When the IRR is calculated, any higher IRR will produce a negative NPV
- This is problematic measure for say tech start-ups with volatile cashflows fluctuating from negative to positive, resulting in multiple IRRs
- Does not account of market rates, typically NPV preferred over this

## Bonds
- They are a fixed income / debt instrument
- Somewhat a securitized loan

### Pricing of Fixed Maturity Bond
- Price of bond : $P_B = \sum _{t=1} ^T \frac{C_t}{(1+r)^t} + \frac{ParValue_T}{(1+r)^T}$

### Pricing of Perpetual Bond (Not Maturity Date)
- Price of perpetuity : $P_B = \sum _{t=1} ^\infty \frac{C_t}{(1+r)^t} = \frac{C}{r}$

### Holding Period Return
- The formula is as follows $H_R = \frac{\sum C + \Delta P_B}{P_0}$
  - $\sum C$: total sum of coupons
  - $\Delta P_B$: difference in price sold and price purchased
  - $P_0$: price paid initially


### Yield Curves
- Typically the longer the bond's maturity, the higher the yield on the bond
- However, there are scenarios where this is not true and we would then have an inverted yield curve
  - Although not definitive, inverted yield curves have often precede equity crashes
- There are many other factors that affect a bond's yield, for example if it has a poor rating similar to junk status, yields are much higher to compensate for the associated higher risks

## Equities

### Equities Valudation: Constant Dividends
- Price of equity : $P_{E} = \sum _{t=1} ^\infty \frac{D_t}{(1+r)^t} = \frac{D}{r}$

### Equities Valudation: Growing Dividends
- Assuming a constant growth of dividend at a rate of $g$
- Price of equity : $P_{E} = \sum _{t=1} ^\infty \frac{D_t (1 + g)^t}{(1+r)^t} = \frac{D}{r - g}$

### Holding Period Return of Equity
- The formula is as follows $H_R = \frac{\sum D + \Delta P_E}{P_0}$
  - $\sum C$: total sum of dividends
  - $\Delta P_B$: difference in price sold and price purchased
  - $P_0$: price paid initially
- Alternatively we can calculate the holding period return using adjusted prices accounting for dividends and splits via $H_R = \frac{\Delta P^{adj}}{P_{0}^{adj}}$

### Stock Volatility
- A stock's volatility can be calculated through $\sigma = \sqrt{\frac{\sum_{t=1} ^T (r_t - \mu)^2}{T}}$

### Primary vs Secondary Market
- Primary market: new shares issued
- Secondary market (NASDAQ etc.): publicly traded shares
- 

