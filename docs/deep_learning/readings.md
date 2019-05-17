!!! info
    This is a list of papers I think are interesting.

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
	
## Network Pruning
- [EigenDamage: Structured Pruning in the Kronecker-Factored Eigenbasis](https://arxiv.org/abs/1905.05934)
    - Compared to existing Hessian-based methods, this works on the KFE
    - Reported 10x reduction in model size and 8x reduction in FLOPs on Wide ResNet32 (WRN32)

## Explainability
- [A Unified Approach to Intepreting Model Predictions](https://arxiv.org/pdf/1705.07874.pdf)
    - Introduces SHAP (SHapley Additive exPlanations)
    - "SHAP assigns each feature an importance value for a particular prediction"
        - Higher positive SHAP values (red) = increase the probability of the class
        - Higher negative SHAP values (blue) = decrease the probability of the class

## Cautious
- [Same Stats, Different Graphs: Generating Datasets with Varied Appearance and Identical Statistics through Simulated Annealing](https://www.autodeskresearch.com/publications/samestats)
    - Shows through scatterplots that multiple toy datasets although visually very different can have similar summary statistics like mean, standard deviation and pearson correlation
    - This paper emphasises the need to always visualize your data