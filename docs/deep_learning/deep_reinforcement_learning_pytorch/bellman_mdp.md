## Markov Decision Processes (MDPs)
- Typically we can **frame all RL tasks as MDPs**
	- Intuitively, it's sort of a way to frame RL tasks such that we can solve them in a "principled" manner. We will go into the specifics throughout this tutorial
- The key in MDPs is the **Markov Property**
	- Essentially the future depends on the present and not the past
		- More specifically, the future is independent of the past given the present
		- There's an assumption the present state encapsulates past information. This is not always true, see the note below.
- Putting into the context of what we have covered so far: our agent can (1) **control its action** based on its current (2) **completely known state**
	- Back to the "driving to avoid puppy" example: given we know there is a dog in front of the car as the current state, the agent can decide to take a left/right turn to avoid colliding with the puppy

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
		- A global minima can somehow be attained via Dynamic Programming(DP)
	2. **Model-free RL**: this is where we ==cannot clearly define== our (1) transition probabilities and/or (2) reward function
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

## Optimal Policy $\pi_*$
- Optimal policy = optimal state-value and action-value functions = max return
	- Essentially we want the maximum expected returns!
- $\pi_{*} = \arg\max_{\pi} \mathcal{V}_{\pi}(s) = \arg\max_{\pi} \mathcal{Q}_{\pi}(s, a)$
