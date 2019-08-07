!!! info
    This is a list of growing number of papers and implementations I think are interesting.

## Long Tailed Recognition
- [Large-Scale Long-Tailed Recognition in an Open World](https://arxiv.org/abs/1904.05160)
    - Frequently in real world scenario there're new unseen classes or samples within the tail classes
    - This tackles the problem with dynamic embedding to bring associative memory to aid prediction of long-tailed classes
    - The model essentially combines direct image features with embeddings from other classes

## Better Generalization (Overfitting Prevention or Regularization)
- [Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907)
    - They propose to use domain randomization to train deep learning algorithms on synthetic data and transferring to real-world data
    - The idea is that with sufficient variability in the textures of synthetic data, real-world data becomes another variation of the synthetic data
    - It works surprisingly well and it's a simple technique of varying image textures essentially enabling CNNs to be more robust to variations in image textures
    
## Optimization
- [Online Learning Rate Adaptation with Hypergradient Descent](https://arxiv.org/abs/1703.04782)
	- Reduces the need for learning rate scheduling for SGD, SGD and nesterov momentum, and Adam
	- Uses the concept of hypergradients (gradients w.r.t. learning rate) obtained via reverse-mode automatic differentiation to dynamically update learning rates in real-time alongside weight updates
	- Little additional computation because just needs just one additional copy of original gradients store in memory
	- Severely under-appreciated paper

## Network Compression 
- [Energy-constrained Compression for Deep Neural Networks via Weighted Sparse Projection and Layer Input Masking](https://arxiv.org/abs/1806.04321)
    - More production applications of DNN require low-energy consumption environment like self-driving cars, VR goggles, and drones
    - As such it's critical to optimize DNN not for its primary performance (accuracy etc.) but for its energy consumption performance too 
    - In the DNN training, this paper introduces an energy budget constraint on top of other optimization objectives
    - This allows optimization of multiple objectives simultaneously (top-1 accuracy and energy consumption for example)
    - It's done through weighted sparse projection and layer input masking
    
## Architecture Search
- [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055)
    - Neural search algorithm based on gradient descent and continuous relaxation in the architecture space. 
    - A good move towards automatic architecture designs of neural networks.
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
    - Scales all dimensions of a CNN, resolution/depth/width using compound coefficient
    - Uses neural architecture search
   
## Network Pruning
- [EigenDamage: Structured Pruning in the Kronecker-Factored Eigenbasis](https://arxiv.org/abs/1905.05934)
    - Compared to existing Hessian-based methods, this works on the KFE
    - Reported 10x reduction in model size and 8x reduction in FLOPs on Wide ResNet32 (WRN32)
    
## Bayesian Deep Learning
- [Fast and Scalable Bayesian Deep Learning by Weight-Perturbation in Adam](https://arxiv.org/abs/1806.04854)
    - Variational Adam (Vadam), an alternative to varianal inference via dropout.
    - Vadam perturbs the network's weights when backpropagating, allowing low computation cost uncertainty estimates. 
    - Not as good as dropout in terms of performance, but a good direction for computationally cheaper options.

## Explainability
- [A Unified Approach to Intepreting Model Predictions](https://arxiv.org/pdf/1705.07874.pdf)
    - Introduces SHAP (SHapley Additive exPlanations)
    - "SHAP assigns each feature an importance value for a particular prediction"
        - Higher positive SHAP values (red) = increase the probability of the class
        - Higher negative SHAP values (blue) = decrease the probability of the class
- [Hierarchical interpretations for neural network predictions](https://openreview.net/pdf?id=SkEqro0ctQ)
    - Given a prediction from the deep neural network, agglomerative contextual decomposition (ACD) produces a hierarchical clusters of input features alongside cluster-wise contribution to the final prediction.
    - The hierarchical clustering is then optimized to identify learned clusters driving the DNN's predictions.

## Cautious
- [Same Stats, Different Graphs: Generating Datasets with Varied Appearance and Identical Statistics through Simulated Annealing](https://www.autodeskresearch.com/publications/samestats)
    - Shows through scatterplots that multiple toy datasets although visually very different can have similar summary statistics like mean, standard deviation and pearson correlation
    - This paper emphasises the need to always visualize your data
    
## Visualization
- [Netron](https://github.com/lutzroeder/netron)
    - Easily visualize your saved deep learning models (PyTorch .pth, TensorFlow .pb, MXNet .model, ONNX, and more)
    - You can even check out each node's documentation quickly in the interface

## Missing Values
- [BRITS](https://arxiv.org/abs/1805.10572)
    - If you face problems in missing data in your time series and you use existing imputation methods, there is an alternative called BRITS where it learns missing values in time series via a bidirectional recurrency dynamical system

## Correlation 
- [DCCA: Deep Canonical Correlation Analysis](http://proceedings.mlr.press/v28/andrew13.pdf)
    - Learn non-linear complex transformations such that resulting transformed data have high linear correlation
    - Alternative to non-parametric methods like kernel canonical correlation analysis (KCCA) and non-linear extension of canonical correlation analysis (CCA)
    - Shown to learn higher correlation representations than CCA/KCCA
        
## Deep Reinforcement Learning
- [An Empirical Analysis of Proximal Policy Optimization with Kronecker-factored Natural Gradients](https://arxiv.org/pdf/1801.05566.pdf)
	- Shows 2 SOTA for deep RL currently (2018 / early 2019): PPO and ACKTR
	- Attempts to combined PPO objective with K-FAC natural gradient optimization: PPOKFAC
	- Does not improve sample complexity, stick with either PPO/ACKTR for now
