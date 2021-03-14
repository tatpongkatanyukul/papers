# Recognition Awareness

This one is to examine the new interpretation of softmax and its implication---LC.

Keys

  * [Softmax](https://github.com/tatpongkatanyukul/papers/blob/main/RecogAwareness/softmax.md)
  * Open-set recognition
  * Open-set recognition evaluation

# 2021 Additional Reviews

## Bridle NIPS 1990

  * propose softmax and analyze that softmax output is approximating conditional probability

> ![E(\theta) = \sum_{t=1}^T \sum_{j=1}^N (Q_j(x_t, \theta) - \delta_{j, c_t})^2](https://latex.codecogs.com/svg.latex?E(\theta)=\sum_{t=1}^T\sum_{j=1}^N(Q_j(x_t,\theta)-\delta_{j,c_t})^2)  
> where ![\delta_{j, c} =1](https://latex.codecogs.com/svg.latex?\delta_{j,c}=1) if j = c, otherwise 0.

That is, ![L(\theta) = \sum_{n=1}^N \sum_{k=1}^K (\hat{y}(x_n, \theta) - y_{n,k})^2](https://latex.codecogs.com/svg.latex?L(\theta)=\sum_{n=1}^N\sum_{k=1}^K(\hat{y}(x_n,\theta)-y_{n,k})^2)



## 
