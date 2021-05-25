# Open Set Recognition

## Neal et al. ECCV 2018: Open Set Learning with Counterfactual Images

Keys I found fascinating
  * Use GAN with reconstruction loss to generate counterfactual images, which later used to train a (K+1)-class classifier
  * The objective to find counterfactual representation zhat is: ```\min_z \| E(x) - z \| + \log( 1 +  \sum_{i=1}^K \exp C_K(G(z))_i)```
  * Wassertein critic: ```\sum_{x \in X} D(G(E(x))) - D(x)```
  * gradient penalty ```\lambda (\| \nabla_x D(x) \| -1 )``` encourages the size of the gradient to be 1.
    * why? see [29]
  * re-parameterization: exploit weights of K-class classifier for (K+1)-class classifier
    * how? see [14]
