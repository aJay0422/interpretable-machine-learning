# Project Description
This project is about identifying important samples in the training dataset using the TracIn score.

# TracIn
[TracIn](https://arxiv.org/abs/2002.08484) is a method to evaluate the influence of a sample $x$ on another sample $x'$.  It can be used to any algorithm that uses gradient descent training process.
$$TracIn(x, x') = \sum_{t: x_t=x} \eta_t \nabla l(w_t, x') \cdot \nabla l(w_t, x)$$  
Here $\eta_t$ is the learning rate at the t-th epoch, $l$ is the loss function and $w_t$ is the model parameter at time t.

# Goal
Our goal is to identify training samples which are influential on test samples, and further achieve the same or higher test accuracy while only train the model on a subset of the training data set.