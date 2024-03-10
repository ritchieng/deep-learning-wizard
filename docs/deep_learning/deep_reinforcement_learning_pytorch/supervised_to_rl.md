---
comments: true
---

# Supervised Learning to Reinforcement Learning

## Supervised Learning
- The tasks we've covered so far fall under the category of supervised learning
	- **Before, we have gone through 2 major tasks: classification and regression with labels**
	- **Classification**: we've a number of MNIST images, we take them as input and we use a neural network for a classification task where we use the ground truth (whether the digits are 0-9) to construct our cross entropy loss
		- [Classification loss](https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_logistic_regression/#cross-entropy-function-d-for-more-than-2-class) function (cross entropy loss): $- \sum^K_1 L_i log(S_i)$
			- $K$: number of classes
			- $L_i$: ground truth (label/target) of i-th class
			- $S_i$: output of softmax for i-th class
	- **Regression**: alternatively we go through a regression task of say, predicting a time-series, but we still have the ground truth we use to construct our loss function
		- [Regression loss](https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_linear_regression/#building-a-linear-regression-model-with-pytorch) function (mean squared error): $\frac{1}{n} \sum_{i=1}^n(\hat y_i - y_i)^2$
			- $\hat{y}$: prediction
			- $y$: ground truth (label/target)
- The key emphasis here is that we have mainly gone through supervised learning tasks that requires labels. Without them, we would not be able to properly construct our loss functions for us to do 2 critical steps (1) [backpropagate to get our gradients](https://www.deeplearningwizard.com/deep_learning/boosting_models_pytorch/derivative_gradient_jacobian/) and (2) [gradient descent to update our weights with our gradients](https://www.deeplearningwizard.com/deep_learning/boosting_models_pytorch/forwardpropagation_backpropagation_gradientdescent/)

!!! note "Loss functions"
	We have covered 2 basic loss functions such as cross entropy loss (classification task) and mean squared error (regression task).

	However there are many more loss functions for both classification and regression task that will be covered in a separate section. 

	For example, there are alternatives to mean squared error (MSE) like mean absolute error (MAE or L1 Loss), Smooth L1 Loss (less sensitive to outliers), quantile regression loss function (allowing confidence intervals) and many more.

## Reinforcement learning is not supervised learning
- One difference is that there is no ground truth (label/target)
	- There is typically no label as to what is the definitively right prediction, we have to explore to find out what's "right" (essentially, the best possible prediction)
- Instead of minimizing a loss function comprising a target and a prediction (as with supervised learning), in reinforcement learning we are typically concerned with maximizing our reward function by trying different actions and exploring what those actions yield in an environment
	- Let's use a simple game example of driving and not colliding with puppies crossing the road
		- **Agent**
			- Driver
		- **Environment**
			- 3 lane road, puppies crossing and the agent
		- **States**
			- Left, center or right lane
		- **Actions**
			- To move from one state to another
			- Turn left, center or right
		- **Reward**
			- Feedback on whether action is good/bad, essentially the goal of the problem
			- Colliding with the puppy: -10 points
			- Too close to the puppy (scares the puppy): -2 points
			- Safe distance from the puppy: 10 points
		- **Value function**
			- Defines what is good in the long-run as compared to rewards which is immediate after an action takes the agent to another state
			- It's somewhat the discounted sum of the rewards the agent is expected to get
		- **Policy**
			- This defines how the agent acts in its states
		- In this case, the **agent** might first collide with the puppy and learn it's bad (-10 points), then try not collide as the second action and still learn it's bad to be too close (-2 points) and finally as the third action learn to steer clear of puppies (+10 points) as it yields the largest **reward**
			- Gradually it'll learn to drive at a safe distance from puppies to collect points (+10 points for safe distance)
			- To do this, the agent needs to go try different actions and learn from its mistakes (trial and error), attempting to maximize its long-term

## 2 Distinguishing Properties of reinforcement learning
- Essentially, 2 distinguishing properties of reinforcement learning are: [^1]
	- (1) "Trial-and-error search"
	- (2) "Delayed reward"


In the next section, we'll be covering the terms we'll dive into these key terms through the lens of Markov Decision Processes (MDPs) and Bellman Equations.

[^1]: Richard S. Sutton and Andrew G. Barto. [Reinforcement Learning: An Introduction; 2nd Edition](http://incompleteideas.net/book/the-book.html). 2017.