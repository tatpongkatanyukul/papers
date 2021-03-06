# Recognition Awareness

This one is to examine the new interpretation of softmax and its implication---LC.

Keys

  * [Softmax](https://github.com/tatpongkatanyukul/papers/blob/main/RecogAwareness/softmax.md)
  * [Open-set recognition](https://github.com/tatpongkatanyukul/papers/blob/main/RecogAwareness/OSR.md)
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

![\hat{y}_k(x)=\sum_{m=1}^M\pi_m\frac{\exp(a_k(x))}{\sum_{i=1}^K\exp(a_i(x))};s.t.\sum_{m=1}^M\pi_m=1](https://latex.codecogs.com/svg.latex?\hat{y}_k(x)=\sum_{m=1}^M\pi_m(x)\frac{\exp(a_k^{(m)}(x))}{\sum_{i=1}^K\exp(a_i^{(m)}(x))};s.t.\sum_{m=1}^M\pi_m(x)=1)

> In other words, MoS computes _M_ Softmax distributions and uses a weighted average of them as the next-token probability distribution.

Parameters _π<sub>m</sub>(x)_ can be learned.

### I found it interesting
"... statistical language modeling has gone through significant development from **traditional Ngram models to neural language model** ..."

"... **as a density estimation problem**, language modeling mostly relies on a universal auto-regressive factorization of the joint probability and then models each conditional factor using different approaches."

"... given a corpus of tokens ```\mathbf{X} = (X_1, ..., X_T)```, the joint probability ```P(\mathbf{X})``` factorizes as ```P(\mathbf{X}) = \prod_t P(X_t|X_{<t}) = \prod_t P(X_t|C_t)```, where ```C_t = X_{<t}``` is referred to as the _context_ of the conditional probability hereafter."

> "... recurrent neural network (RNN) based language models achieve state-of-the-art results on various benchmarks ... "

> "... A standard approach is to use a recurrent neural network to **encode the context into a fixed sized vector**, which is tehn multiplied by the word embeddings using dot product to obtain the logits. The logits are consumed by **the Softmax function to give a categorical probability distribution over the next token.**"

> "... **an unclear question is whether the combination of dot product and Softmax is capable of modeling the conditional probability, which can vary dramatically with the change of the context.**"

> "... natural language is highly context-dependent, the matrix to be factorized can be high-rank. 
> This further implies that standard Softmax-based language models with distributed (output) word embedding do not have enough capacity to model natural language. We call this the **Softmax bottleneck**"

### Synopsis
  * 1. logit = argument of the exponential in a softmax (it is penultimate if the softmax is the last layer)
  * 2. "... when the dimension _d_ is too small, Softmax does not have the capacity to express the true data distribution."
    * _d_     
      * next-token probability ```P_\theta(x|c)``` is modeled by Softmax: ```P_\theta(x|c) = \exp h_c^T w_x / \sum_{x'} \exp h_c^T w_x'``` where context vector ```h_c``` is a function of context _c_ and word embedding ```w_x``` is a function of next-token _x_. 
      * Both ```h_c``` and ```w_x``` are parameterized by ```\theta``` and have the same dimension _d_
      
## Kamai et al. NIPS 2018

Kamai et al. propose sigsoftmax as a mitigation to softmax bottleneck issue. They claim and show that gradient calculation of sigsoftmax is numerically stable.

Kamai et al discuss desirable properties of the classification output function
  * Non-negative
  * Monotonically increasing
  * Numerically stable

> In training of deep learning, we need to calculate the gradient for optimization. The derivative of logarithm of [f(z)]<sub>i</sub> with respect to _z<sub>j</sub>_ is

![\frac{d\log([y(z)]_i)}{dz_j}=\frac{1}{[y(z)]_i}\frac{d[y(z)]_i}{dz_j}](https://latex.codecogs.com/svg.latex?\frac{d\log([y(z)]_i)}{dz_j}=\frac{1}{[y(z)]_i}\frac{d[y(z)]_i}{dz_j})

  * Nonlinearity of log(y(x))

Kamai et al propose sigsoftmax:

![[f(z)]_i=\frac{\exp(z_i)\sigma(z_i)}{\sum_{m=1}^M\exp(z_m)\sigma(z_m)}](https://latex.codecogs.com/svg.latex?[f(z)]_i=\frac{\exp(z_i)\sigma(z_i)}{\sum_{m=1}^M\exp(z_m)\sigma(z_m)})

> "_[log(y(x))]_ should be nonlinear."

[Detail analysis of property 2](https://github.com/tatpongkatanyukul/papers/blob/main/RecogAwareness/rsc/r01sigsoftmax.pdf)

### Idea!

Instead of sigsoftmax, use softplusmax

Softplus: ***sp(a) = log(1 + exp(a))***


Softplusmax:

![[f(z)]_i=\frac{sp(z_i)}{\sum_{m=1}^Msp(z_m)}](https://latex.codecogs.com/svg.latex?[f(z)]_i=\frac{sp(z_i)}{\sum_{m=1}^Msp(z_m)})

```R
xs = seq(-10, 10, 0.001)
softplus = function(x){log(1 + exp(x))}
plot(xs, softplus(xs), type='l')
```

Kamai et al's properties:
 * Nonlinearity of _log(g(a))_: _ _log(log(1 + exp(a)))_ 
 ```R
 plot(xs, log(softplus(xs)), type='l', col='red')
 ```
 * Numerically stable? **Tired! BREAK HERE**: **d log y<sub>i</sub> / d a<sub>j</sub>** = ???
 * Non-negative: sp(a) >= 0.
 * Monotonically increasing: a<sub>1</sub> <= a<sub>2</sub> implies  _log(g(a<sub>1</sub>)) <= log(g(a<sub>2</sub>))_

## Memisevic et al's Gated Softmax Classification. NIPS 2010

My verdict
  * I don't understand its mechanism how it works. Especially, how do they get the hidden variable values or how they train their model to get them

I don't quite get the key contribution of the paper. It may be:
> ... allows the model encode ***invariances*** inherent in a task by learning a dictionary of ***invariant basis functions***.

> If each training image was labeled with both the class and the values of a set binary style features, it would make sense to use the image features to create a bipartite conditional random field (CRF) which gave low energy to combinations of a class label and a style feature that were compatible with the image feature.
> This would force the in which local features were interpreted to be globally consistent about style features such as stroke thickness or "italicness". But what if the values of the style features are missing from the training data?
> We describe ***a way of learning a large set of binary style features from training data that are only labeled with the class.***
> Our "gated softmax" model allows the 2<sup>K</sup> possible combinations of the K learned style features to be integrated out.

> We introduce a vector ***h*** of _binary latent variables (h<sub>1</sub>, ..., h<sub>K</sub>)_ and replace the linear score with a bilinear score of ***x*** and ***h***:
 
> s<sub>y</sub>(***x***, ***h***) = ***h***<sup>T</sup> W<sub>y</sub> ***x***.

> p(y, ***h***|***x***) = exp(***h***<sup>T</sup> W<sub>y</sub> ***x***)/ Σ<sub>y' ***h'***</sub> exp(***h***<sup>T</sup> W<sub>y</sub> ***x***)

Output y can be perceived as a class, while ***h*** is perceived as a style (or a kind of a latent characteristic).

Variable _W<sub>y</sub>_ of size K x D is a weight set for class _y_, where K is a number of latent dimentions (kinda # of style dimensions)
and D is a number of input dimentions: ***x*** = [x<sub>1</sub>, ..., x<sub>D</sub>].
The matrix _W<sub>y</sub>_ = [_w<sub>yik</sub>_].

> p(y|***x***) = Σ<sub>***h***</sub> p(y, ***h***|***x***) = p(y, ***h***=[0,0,...,0,0]<sup>T</sup>|***x***) + p(y, ***h***=[0,0,...,0,1]<sup>T</sup>|***x***) + ... + p(y, ***h***=[1,1,...,1,1]<sup>T</sup>|***x***)

> In order to obtain class-invariant features, we factorize the parameter tensor _W_ as follows:

w<sub>yik</sub> = Σ<sub>f=1</sub><sup>F</sup> w<sub>if</sub><sup>x</sup> w<sub>yf</sub><sup>y</sup> w<sub>kf</sub><sup>h</sup>

> The model parameters are now given by three matrices W<sup>x</sup>, W<sup>y</sup>, W<sup>h</sup>, and each component _W<sub>yik</sub>_ of _W_ is defined as a ***three-way inner product*** of column vectors taken from these matrices.
> This factorization of a ***three-way parameter tensor*** was previously used by [3: Memisevic Hinton's Learning to represent spatial transformations with factored higher-order Boltzmann machines. Neural Computation 2010] to reduce the number of parameters in an unsupervised model of images.

> Our model gets its power from the fact that inputs, hidden variables and labels interact in ***three-way cliques***. ***Factored three-way interactions*** make it possible to learn task-specific features and to learn ***transformational invariances*** inherent in the task at hand.

> ... Encourage the hidden unit activities to be sparse (e.g. using approach in [20: Lee et al's Sparse deep belief net model for visual area V2 NIPS 2007](https://papers.nips.cc/paper/2007/hash/4daa3db355ef2b0e64b472968cb70f0d-Abstract.html)) and/or traing the model semi-supervised are further directions for further research.


[CODE](http://www.cs.toronto.edu/~rfm/gatedsoftmax/index.html)

> Say, you trained K class-specific Restricted Boltzmann Machines and you would like to combine the K RBMs for classification. Unlike with, say, mixtures of Gaussians, you cannot simply use Bayes' rule, because each RBM will have a different partition function.

> As it turns, however, there is in fact a principled way to combine the RBMS for classification:

> Just think of the set of K class-specific RBMs as a single conditional distribution p(inputs, hiddens|class). Now compute p(class|inputs), integrating over the hiddens. The partition functions cancel and you can compute both the probability and the derivatives with respect to all the RBMs parameters in polynomial time. This is the "gated softmax classifier"


### IDEA!!! (Big / Big Goal leads to Significant Findings!)

How can we utilize multi-aspect labels in ML? 
It may be unsupervised/semi-supervised/transfer learning. But, how to effectively/systematically do it.
It's a big goal. We need a big goal for a potentially big finding.

A data can have multiple aspects, e.g., classes and styles.
Some training samples may have class labels, but not style labels.
Some may have style labels without class labels.
Some may have both. Some may have none.

It is rather classic endeavor that how we can utilize all of these.

This can be beneficial to the coming engineering/medicine collaborative projects, as well.
Faculty of Medicine has larget sets of data, but not labels.

## Neal et al's Open Set Learning with Counterfactual Images ECCV 2018

> "Our approach, based on generative adversarial neworks, **generates examples that are close to training set examples yet do not belong to any training category.** ... **reformulate open set recognition as classification with one additional class**"

> "Fig. 1. ... Given known examples we generate counterfactual examplesfor the unknown class. The decision boundary between known and counterfactual unknown examples extends to unknown examples, similar to the idea that one can train an SVM with only support vectors.

## Yoshihashi et al's Classification-Reconstruction Learning for Open-Set Recognition CVPR 2019

> "Open-set classification is a problem of handling 'unknown' classes that are not contained in the training dataset, whereas traditional clasifiers assume that only known clases appear in the test environment."

> "Existing open-set classifiers rely on **deep networks trained in a supervised manner on known classes in the training set; this causes specialization of learned representations to known classes and makes it hard** to distinguish unknowns from knowns."
> "In contrast, we train networks for joint classification and reconstruction of input data. This enhances the learned representation so as to **preserve information** useful for separating unknowns from knowns."

> "To be deployable to real applications, recognition systems need to be tolerant of unknown things and events that were not anticipated during the training phase. However, most of the existing learning methods are based on the closed-world assumption, that is, the training datasets are assumed to include all classes that appear in the environments where the system will be deployed. This assumption can be easily violated in real-world problems, where covering all possible classes is almost impossible. **Closed-set classifiers are error-prone to samples of unknown classes, and this limits their usability.**"

> "For features to represent the samples, almost all existing deep open-set classifiers rely on those acquired via fully supervised learning, as shown in Fig. 1 (a). However, they are for emphasizing **the discriminative features of known classes; they are not necessarily useful for representing unknowns or separating unknowns from knowns.**"

![YoshihashiEtAl Fig. 1](https://github.com/tatpongkatanyukul/papers/raw/main/RecogAwareness/rsc/YoshihashiFig1.png)

> "Regarding the representations of outliers that we cannot assume beforehand, it is natural to **add unsupervised learning as a regularizer so that the learned representaions acquire information that are important in general but may not be useful for classifying given classes.**"

> "Reconstruction of input samples from low-dimensional latent representations inside the networks is a general way of unsupervised learning. The representation learned via reconstruction are useful in several tasks [Zhang et al ICML 2016].

> "While the known-class classifier exploits supervisedly learned prediction y, the unknown detector uses a reconstructive latent representation z together with y. This allows unknown detectors to exploit a wider pool of featuresthat may not be discriminative for known classes. Additionally, **in higher-level layers of supervised deep nets, details of input tend to be lost, which may not be preferable in unknown detection**.

> ... the key idea in DHRNets i the bottlenecked lateral connections, which is useful to learn rich representations for classification and compact representations for detection of unknowns jointly.

> ... **This bottlenecking is crucial**, because outliers are harder to detect in higher dimensional feature spaces due to _concentration on the sphere_[Zimek et al's A survey on unsupervised outlier detection in high-dimensional numerical data 2012]. Existing autoencoder variants, which are useful for outlier detection by learning compact representation, cannot afford large-scale classification because the bottlenecks in their mainstreams limit the expressive power for classification.

### IDEA !

Learn latent representation with high dimension, but enforcing sparsity

## Yolov3 

use multiple binary outputs, allowing multi-aspect or hierarchical classification

## Facenet

separate feature vectors from the final classification
