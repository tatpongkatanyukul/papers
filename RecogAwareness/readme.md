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

> where ***δ<sub>j,c</sub>=1*** if ***j = c***, otherwise ***0***.

That is, 

![L(\theta) = \sum_{n=1}^N \sum_{k=1}^K (\hat{y}(x_n, \theta) - y_{n,k})^2](https://latex.codecogs.com/svg.latex?L(\theta)=\sum_{n=1}^N\sum_{k=1}^K(\hat{y}_k(x_n,\theta)-y_{n,k})^2)

> "It is well known that the value of ***F(x)*** which minimises the expected value of ***(F(x) - y)<sup>2</sup>*** is the expected value of ***y*** given ***x***. The expected value of ***δ<sub>j,c<sub>t</sub></sub>*** is ***P(C=j|X=x<sub>t</sub>)***, the probability that the class associated with ***x<sub>t</sub>*** is the ***j<sup>th</sup>*** class."

That is,

![argmin_{F(x)} E[(F(x) - y)^2] = E[y|x]](https://latex.codecogs.com/svg.latex?argmin_{F(x)}E[(F(x)-y)^2]=E[y|x])

and ***E[y<sub>k</sub>|x] = P(C=k|X=x)***.

> Maximum likelihood (ML) training is appropriate if we are choosing from a family of pdfs which includes the correct one. In most real-life applications of pattern classification we do not have knowledge of the form of the data distributions, although we may have some useful ideas. In that case ML may be a rather bad approach to pdf estimation for the _purpose of pattern classification_, because what matters is the _relative densities_.

> An alternative is to optimise a measure of success in pattern classification, and this can make a big difference to performance, particularly when the assumptions about the form of the class pdfs is badly wrong.

> ... For discrimination training of sets of stochastic models, Bahl et al. suggest maximising the Mutual Information, I, between the training observations and the choice of the corresponding correct class.

Mutual Information (MI) of the joint events:

![I(X,Y)=\sum_{(x,y)} P(X=x,Y=y)\log \frac{P(X=x,Y=y)}{P(X=x)P(Y=y)}](https://latex.codecogs.com/svg.latex?I(X,Y)=\sum_{(x,y)}P(X=x,Y=y)\log\frac{P(X=x,Y=y)}{P(X=x)P(Y=y)})

> ... is equvalent, to minimise minus its log:

> ![J=-\sum_{t=1}^T \log Q_{c_t}(x_t)](https://latex.codecogs.com/svg.latex?J=-\sum_{t=1}^T\log(Q_{c_t}(x_t)))

That is "cross entropy loss":

![J=-\sum_{n=1}^N \log \hat{y}_{c_n}(x_n)](https://latex.codecogs.com/svg.latex?J=-\sum_{n=1}^N\log(\hat{y}_{c_n}(x_n)))

where ***c<sub>n</sub>*** is a correct class of the ***n<sup>th</sup>*** sample.


## Yang et al. ICLR 2018

  * show that "the expressiveness of Softmax-based models (including the majority of neural language models) is limited by a Softmax bottleneck."
  * propose Mixture of Softmax (MoS) to mitigate the issue

Given **logit** ***h<sub>c</sub><sup>T</sup> w<sub>x</sub>*** (i.e., "penultimate" a = h<sub>c</sub><sup>T</sup> w<sub>x</sub>),

> the model distribution is usually written as 

> ***P<sub>θ</sub>(x|c) = exp h<sub>c</sub><sup>T</sup> w<sub>x</sub> / Σ<sub>x'</sub> exp h<sub>c</sub><sup>T</sup> w<sub>x</sub>***

> where ***h<sub>c</sub>*** is a function of _c_, and ***w<sub>x</sub>*** is a function of _x_.

Yang et al. work on NLP and context _c_ plays an input role, while a word _x_ plays an output.

Given that _N_ is a number of the target classes and
_M_ is a number of samples,

> ***H<sub>θ</sub>*** = [h<sub>c<sub>1</sub></sub><sup>T</sup>; h<sub>c<sub>2</sub></sub><sup>T</sup>; ...; h<sub>c<sub>N</sub></sub><sup>T</sup>;]

> ***W<sub>θ</sub>*** = [w<sub>x<sub>1</sub></sub><sup>T</sup>; w<sub>x<sub>1</sub></sub><sup>T</sup>; ...; w<sub>x<sub>M</sub></sub><sup>T</sup>;] 

> ***A*** = [log P<sup>*</sup>(x<sub>1</sub>|c<sub>1</sub>), log P<sup>*</sup>(x<sub>2</sub>|c<sub>1</sub>), ..., log P<sup>*</sup>(x<sub>M</sub>|c<sub>1</sub>); ... log P<sup>*</sup>(x<sub>M</sub>|c<sub>N</sub>)]

> where ***H<sub>θ</sub>*** in R<sup>N x d</sup>, ***W<sub>θ</sub>*** in R<sup>M x d</sup>, ...

_d_ is a number of hidden dimensions of the last feature vector.

Yang et al. then use linear algebra to deduce that if ***d < rank(A) - 1*** (see Yang et al's Corollary 1), then there is a context _c_ in language such that ***P<sub>θ</sub>(X|c) ≠ P(X|c)***.

That is, when a number of dimensions of the last feature vector is less than a rank of output matrix, the softmax output is less effective as an approximator of the class conditional probability.

Yang et al. propose mixture of softmaxes:

![\hat{y}_k(x)=\sum_{m=1}^M\pi_m\cdot\frac{\exp(a_k(x))}{\sum_{i=1}^K\exp(a_i(x))};s.t.\sum_{m=1}^M\pi_m=1](https://latex.codecogs.com/svg.latex?\hat{y}_k(x)=\sum_{m=1}^M\pi_m\cdot\frac{\exp(a_k(x))}{\sum_{i=1}^K\exp(a_i(x))};s.t.\sum_{m=1}^M\pi_m=1)





