# Open Set Recognition

## Neal et al. ECCV 2018: Open Set Learning with Counterfactual Images

Keys I found fascinating
  * Use GAN with reconstruction loss to generate counterfactual images, which later used to train a (K+1)-class classifier
  * The objective to find counterfactual representation zhat is: ```\min_z \| E(x) - z \| + \log( 1 +  \sum_{i=1}^K \exp C_K(G(z))_i)```
  * Wassertein critic: ```\sum_{x \in X} D(G(E(x))) - D(x)```
  * gradient penalty ```\lambda (\| \nabla_x D(x) \| -1 )``` encourages the size of the gradient to be 1.
    * why? see [29] [Gulrajani et al. Improved training of wasserstein gans. 2017.](https://papers.nips.cc/paper/2017/hash/892c3b1c6dccd52936e27cbd0ff683d6-Abstract.html)
  * re-parameterization: exploit weights of K-class classifier for (K+1)-class classifier
    * how? see [14] [Salimans et al. Improved techniques for training gans. NIPS 2016.](https://papers.nips.cc/paper/2016/hash/8a3363abe792db2d8761d6403605aeb7-Abstract.html)
