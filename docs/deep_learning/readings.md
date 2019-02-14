## Deep Reinforcement Learning
- [An Empirical Analysis of Proximal Policy Optimization with Kronecker-factored Natural Gradients](https://arxiv.org/pdf/1801.05566.pdf)
	- Shows 2 SOTA for deep RL currently (2018 / early 2019): PPO and ACKTR
	- Attempts to combined PPO objective with K-FAC natural gradient optimization: PPOKFAC
	- Does not improve sample complexity, stick with either PPO/ACKTR for now

## Optimization
- [Online Learning Rate Adaptation with Hypergradient Descent](https://arxiv.org/abs/1703.04782)
	- Reduces the need for learning rate scheduling for SGD, SGD and nesterov momentum, and Adam,
	- Uses the concept of hypergradients (gradients w.r.t. learning rate) obtained via reverse-mode automatic differentiation to dynamically update learning rates in real-time alongside weight updates
	- Little additional computation because just needs just one additional copy of original gradients store in memory
	- Severely under-appreciated paper