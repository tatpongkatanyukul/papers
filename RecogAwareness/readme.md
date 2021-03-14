# Recognition Awareness

This one is to examine the new interpretation of softmax and its implication---LC.

Keys

  * [Softmax](https://github.com/tatpongkatanyukul/papers/blob/main/RecogAwareness/softmax.md)
  * Open-set recognition
  * Open-set recognition evaluation

# 2021 Additional Reviews

## Bridle NIPS 1990

  * propose softmax and analyze that softmax output is approximating conditional probability
  * show relationship between the classification network and maximum mutual information

> ![Q_j(x) = e^{V_j(x)}/ \sum_k e^{V_k(x)}](https://latex.codecogs.com/svg.latex?Q_j(x)=e^{V_j(x)}/\sum_ke^{V_k(x)})

That is a softmax,

> ![\hat{y}_j(x) = e^{a_j(x)}/ \sum_k e^{a_k(x)}](https://latex.codecogs.com/svg.latex?\hat{y}_j(x)=e^{a_j(x)}/\sum_ke^{a_k(x)})

> A common procedure is to minimise ***E(θ)***, the sum of the squares of the differences between the network outputs and true class indicators, or targets:
> ![E(\theta) = \sum_{t=1}^T \sum_{j=1}^N (Q_j(x_t, \theta) - \delta_{j, c_t})^2](https://latex.codecogs.com/svg.latex?E(\theta)=\sum_{t=1}^T\sum_{j=1}^N(Q_j(x_t,\theta)-\delta_{j,c_t})^2)  
> where ![\delta_{j, c} =1](https://latex.codecogs.com/svg.latex?\delta_{j,c}=1) if j = c, otherwise 0.

That is, 

![L(\theta) = \sum_{n=1}^N \sum_{k=1}^K (\hat{y}(x_n, \theta) - y_{n,k})^2](https://latex.codecogs.com/svg.latex?L(\theta)=\sum_{n=1}^N\sum_{k=1}^K(\hat{y}_k(x_n,\theta)-y_{n,k})^2)

> "It is well known that the value of ***F(x)*** which minimises the expected value of ***(F(x) - y)<sup>2</sup>*** is the expected value of ***y*** given ***x***. The expected value of ***δ<sub>j,c<sub>t</sub></sub>*** is ***P(C=j|X=x<sub>t</sub>)***, the probability that the class associated with ***x<sub>t</sub>*** is the ***j<sup>th</sup>*** class."

That is,

![argmin_{F(x)} E[(F(x) - y)^2] = E[y|x]](https://latex.codecogs.com/svg.latex?argmin_{F(x)}E[(F(x)-y)^2]=E[y|x])

and

![E[y_k|x] = P(C=k|X=x)](https://latex.codecogs.com/svg.latex?E[y_k|x]=P(C=k|X=x))

![E[y_k|x] = P(C=k|X=x)](https://latex.codecogs.com/svg.latex?E[y_k|x]=P(C=k|X=x))



## 
