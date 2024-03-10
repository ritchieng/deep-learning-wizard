---
comments: true
---

## Markov Decision Processes (MDPs)
- Typically we can **frame all RL tasks as MDPs**[^1]
	- Intuitively, it's sort of a way to frame RL tasks such that we can solve them in a "principled" manner. We will go into the specifics throughout this tutorial
- The key in MDPs is the **Markov Property**
	- Essentially the future depends on the present and not the past
		- More specifically, the future is independent of the past given the present
		- There's an assumption the present state encapsulates past information. This is not always true, see the note below.
- Putting into the context of what we have covered so far: our agent can (1) **control its action** based on its current (2) **completely known state**
	- Back to the "driving to avoid puppy" example: given we know there is a dog in front of the car as the current state and the car is always moving forward (no reverse driving), the agent can decide to take a left/right turn to avoid colliding with the puppy in front

## Two main characteristics for MDPs
1. Control over state transitions
2. States completely observable

!!! note "Other Markov Models"
	Permutations of whether there is presence of the two main characteristics would lead to different Markov models. These are not important now, but it gives you an idea of what other frameworks we can use besides MDPs.

	###### Types of Markov Models
	1. {++Control++} over state transitions and {++completely observable++} states: **MDPs**
	2. {++Control++} over state transitions and ==partially observable states==: **Partially Observable MDPs (POMDPs)**
	3. ==No control== over state transitions and {++completely observable++} states: **Markov Chain**
	4. ==No control== over state transitions and ==partially observable== states: **Hidden Markov Model**

	###### POMDPs
	- Imagine our driving example where we don't know if the car is going forward/backward in its state, but only know there is a puppy in the center lane in front, this is a partially observable state
	- There are ways to counter this
		1. Use the **complete history** to construct the current state
		2. Represent the current state as a probability distribution (**bayesian approach**) of what the agent perceives of the current state
		3. Using a **RNN** to form the current state that encapsulates the past[^2]

## 5 Components of MDPs
1. $\mathcal{S}$: set of states
2. $\mathcal{A}$: set of actions
3. $\mathcal{R}$: reward function
4. $\mathcal{P}$: transition probability function
4. $\gamma$: discount for future rewards

!!! tip "Remembering 5 components with a mnemonic"
	A mnemonic I use to remember the 5 components is the acronym "SARPY" (sar-py). I know $\gamma$ is not $\mathcal{Y}$ but it looks like a `y` so there's that.

## Moving From MDPs to Optimal Policy
- We have an **agent** acting in an **environment**
- The way the **environment** reacts to the agent's **actions ($a$)** is dictated by a **model**
- The agent can take **actions ($a$)** to move from one **state ($s$)** to another **new state ($s')$**
- When the agent has transited to a **new state ($s')$**, there will a **reward ($r$)**
- We may or may not know our **model**
	1. **Model-based RL**: this is where we can {++clearly define++} our (1) transition probabilities and/or (2) reward function
		- A global minima can be attained via Dynamic Programming (DP)
	2. **Model-free RL**: this is where we ==cannot clearly define== our (1) transition probabilities and/or (2) reward function
		- Most real-world problems are under this category so we will mostly place our attention on this category
- How the agent **acts ($a$)** in its current **state ($s$)** is specified by its **policy ($\pi(s)$)**
	- It can either be deterministic or stochastic
		1. **Deterministic policy**: $a = \pi(s)$
		2. **Stochastic policy**: $\mathbb{P}_\pi [A=a \vert S=s] = \pi(a | s)$
		 	- This is the proability of taking an action given the current state under the policy
			- $\mathcal{A}$: set of all actions
			- $\mathcal{S}$: set of all states
- When the agent acts given its state under the **policy ($\pi(a | s)$)**, the **transition probability function $\mathcal{P}$** determines the subsequent **state ($s'$)**
	- $\mathcal{P}_{ss'}^a = \mathcal{P}(s' \vert s, a)  = \mathbb{P} [S_{t+1} = s' \vert S_t = s, A_t = a]$
- When the agent act based on its **policy ($\pi(a | s)$)** and transited to the new state determined by the transition probability function $\mathcal{P}_{ss'}^a$ it gets a reward based on the **reward function** as a feedback
	- $\mathcal{R}_s^a = \mathbb{E} [\mathcal{R}_{t+1} \vert S_t = s, A_t = a]$
- Rewards are short-term, given as feedback after the agent takes an action and transits to a new state. Summing all future rewards and discounting them would lead to our **return $\mathcal{G}$**
	- $\mathcal{G}_t = \sum_{i=0}^{N} \gamma^k \mathcal{R}_{t+1+i}$
		- $\gamma$, our discount factor which ranges from 0 to 1 (inclusive) reduces the weightage of future rewards allowing us to balance between short-term and long-term goals
- With our **return $\mathcal{G}$**, we then have our **state-value function $\mathcal{V}_{\pi}$** (how good to stay in that state) and our **action-value or q-value function $\mathcal{Q}_{\pi}$** (how good to take the action)
	- $\mathcal{V}_{\pi}(s) = \mathbb{E}_{\pi}[\mathcal{G}_t \vert \mathcal{S}_t = s]$
	- $\mathcal{Q}_{\pi}(s, a) = \mathbb{E}_{\pi}[\mathcal{G}_t \vert \mathcal{S}_t = s, \mathcal{A}_t = a]$
	- The advantage function is simply the difference between the two functions $\mathcal{A}_{\pi}(s, a) = \mathcal{Q}_{\pi}(s, a) - \mathcal{V}_{\pi}(s)$
		- Seems useless at this stage, but this advantage function will be used in some key algorithms we are covering
- Since our policy determines how our agent acts given its state, achieving an **optimal policy $\pi_*$** would mean achieving optimal actions that is exactly what we want!

!!! note "Basic Categories of Approaches"
	We've covered state-value functions, action-value functions, model-free RL and model-based RL. They form general overarching categories of how we design our agent.

	#### Categories of Design
	1. State-value based: search for the optimal state-value function (goodness of action in the state)
	2. Action-value based: search for the optimal action-value function (goodness of policy)
	3. Actor-critic based: using both state-value and action-value function 
	4. Model based: attempts to model the environment to find the best policy
	5. Model-free based: trial and error to optimize for the best policy to get the most rewards instead of modelling the environment explicitly

## Optimal Policy
- Optimal policy $\pi_*$ --> optimal state-value and action-value functions --> max return --> argmax of value functions
	- $\pi_{*} = \arg\max_{\pi} \mathcal{V}_{\pi}(s) = \arg\max_{\pi} \mathcal{Q}_{\pi}(s, a)$
- To calculate argmax of value functions --> we need max return $\mathcal{G}_t$ --> need max sum of rewards $\mathcal{R}_s^a$
- To get max sum of rewards $\mathcal{R}_s^a$ we will rely on the Bellman Equations.[^3]


## Bellman Equation
- Essentially, the Bellman Equation breaks down our value functions into two parts
	1. Immediate reward
	2. Discounted future value function
- State-value function can be broken into:
	- $\begin{aligned}
		\mathcal{V}_{\pi}(s) &= \mathbb{E}[\mathcal{G}_t \vert \mathcal{S}_t = s] \\
		&= \mathbb{E} [\mathcal{R}_{t+1} + \gamma \mathcal{R}_{t+2} + \gamma^2 \mathcal{R}_{t+3} + \dots \vert \mathcal{S}_t = s] \\
		&= \mathbb{E} [\mathcal{R}_{t+1} + \gamma (\mathcal{R}_{t+2} + \gamma \mathcal{R}_{t+3} + \dots) \vert \mathcal{S}_t = s] \\
		&= \mathbb{E} [\mathcal{R}_{t+1} + \gamma \mathcal{G}_{t+1} \vert \mathcal{S}_t = s] \\
		&= \mathbb{E} [\mathcal{R}_{t+1} + \gamma \mathcal{V}_{\pi}(\mathcal{s}_{t+1}) \vert \mathcal{S}_t = s]
		\end{aligned}$
- Action-value function can be broken into:
	- $\mathcal{Q}_{\pi}(s, a) = \mathbb{E} [\mathcal{R}_{t+1} + \gamma \mathcal{Q}_{\pi}(\mathcal{s}_{t+1}, \mathcal{a}_{t+1}) \vert \mathcal{S}_t = s, \mathcal{A} = a]$


## Key Recap on Value Functions
- $\mathcal{V}_{\pi}(s) = \mathbb{E}_{\pi}[\mathcal{G}_t \vert \mathcal{S}_t = s]$
	- State-value function: tells us how good to be in that state
- $\mathcal{Q}_{\pi}(s, a) = \mathbb{E}_{\pi}[\mathcal{G}_t \vert \mathcal{S}_t = s, \mathcal{A}_t = a]$
	- Action-value function: tells us how good to take actions given state


## Bellman Expectation Equations
- Now we can move from Bellman Equations into Bellman Expectation Equations
- **Basic: State-value function $\mathcal{V}_{\pi}(s)$**
	1. Current state $\mathcal{S}$
	2. Multiple possible actions determined by stochastic policy $\pi(a | s)$
	3. Each possible action is associated with a action-value function $\mathcal{Q}_{\pi}(s, a)$ returning a value of that particular action
	4. Multiplying the possible actions with the action-value function and summing them gives us an indication of how good it is to be in that state
		- **Final equation: $\mathcal{V}_{\pi}(s) = \sum_{a \in \mathcal{A}} \pi(a | s) \mathcal{Q}(s, a)$**
		- **Loose intuitive interpretation**: 
			- state-value = sum(policy determining actions * respective action-values)
- **Basic: Action-value function $\mathcal{Q}_{\pi}(s, a)$**
	1. With a list of possible multiple actions, there is a list of possible subsequent states $s'$ associated with:
		1. state value function $\mathcal{V}_{\pi}(s')$ 
		2. transition probability function $\mathcal{P}_{ss'}^a$ determining where the agent could land in based on the action
		3. reward $\mathcal{R}_s^a$ for taking the action
	2. Summing the reward and the transition probability function associated with the state-value function gives us an indication of how good it is to take the actions given our state
		- **Final equation: $\mathcal{Q}_{\pi}(s, a) = \mathcal{R}_s^a + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a {V}_{\pi}(s')$**
		- **Loose intuitive interpretation**: 
			- action-value = reward + sum(transition outcomes determining states * respective state-values)
- **Expanded functions (substitution)**
	- Substituting action-value function into the **state-value function**
		- $\mathcal{V}_{\pi}(s) = \sum_{a \in \mathcal{A}} \pi(a | s) (\mathcal{R}_s^a + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a {V}_{\pi}(s'))$
	- Substituting  state-value function into **action-value function**
		- $\mathcal{Q}_{\pi}(s, a) = \mathcal{R}_s^a + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a \sum_{a' \in \mathcal{A}} \pi(a' | s') \mathcal{Q}(s', a')$

## Bellman Optimality Equations
- Remember optimal policy $\pi_*$ --> optimal state-value and action-value functions --> argmax of value functions
	- $\pi_{*} = \arg\max_{\pi} \mathcal{V}_{\pi}(s) = \arg\max_{\pi} \mathcal{Q}_{\pi}(s, a)$
- Finally with Bellman Expectation Equations derived from Bellman Equations, we can derive the equations for the argmax of our value functions
- **Optimal state-value function**
	- $\mathcal{V}_*(s) = \arg\max_{\pi} \mathcal{V}_{\pi}(s)$
	- Given $\mathcal{V}_{\pi}(s) = \sum_{a \in \mathcal{A}} \pi(a | s) (\mathcal{R}_s^a + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a {V}_{\pi}(s'))$
	- We have $\mathcal{V}_*(s) = \max_{a \in \mathcal{A}} (\mathcal{R}_s^a + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a {V}_{*}(s')))$
- **Optimal action-value function**
	- $\mathcal{Q}_*(s) = \arg\max_{\pi} \mathcal{Q}_{\pi}(s)$
	- Given $\mathcal{Q}_{\pi}(s, a) = \mathcal{R}_s^a + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a \sum_{a' \in \mathcal{A}} \pi(a' | s') \mathcal{Q}(s', a')$
	- We have $\mathcal{Q}_{*}(s, a) = \mathcal{R}_s^a + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a max_{a' \in \mathcal{A}} \mathcal{Q}_{*}(s', a')$


## Optimal Action-value and State-value functions
1. If the entire environment is known, such that we know our reward function and transition probability function, then we can solve for the optimal action-value and state-value functions via **Dynamic Programming** like
	- Policy evaluation, policy improvement, and policy iteration
2. However, typically we don't know the environment entirely then there is not closed form solution in getting optimal action-value and state-value functions. Hence, we need other iterative approaches like
	1. **Monte-Carlo methods**
	2. **Temporal difference learning** (model-free and learns with episodes)
		1. On-policy TD: SARSA
		2. Off-policy TD: Q-Learning and Deep Q-Learning (DQN)
	3. **Policy gradient**
		- REINFORCE
		- Actor-Critic
		- A2C/A3C
		- ACKTR
		- PPO
		- DPG
		- DDPG (DQN + DPG)

!!! note "Closed form solution"
	If there is a closed form solution, then the variables' values can be obtained with a finite number of mathematical operations (for example add, subtract, divide, and multiply).

	For example, solving $2x = 8 - 6x$ would yield $8x = 8$ by adding $6x$ on both sides of the equation and finally yielding the value of $x=1$ by dividing both sides of the equation by $8$. 

	These finite 2 steps of mathematical operations allowed us to solve for the value of x as the equation has a closed-form solution.

	However, many cases in deep learning and reinforcement learning there are no closed-form solutions which requires all the iterative methods mentioned above.

[^1]: Bellman, R. A Markovian Decision Process. Journal of Mathematics and Mechanics. 1957.
[^2]: Matthew J. Hausknecht and Peter Stone. [Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/abs/1507.06527). 2015.
[^3]: R Bellman. On the Theory of Dynamic Programming. Proceedings of the National Academy of Sciences. 1952.


