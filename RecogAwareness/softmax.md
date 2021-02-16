# Softmax

I feel like I should have emphasized more on the softmax.

Here's what I have found on the softmax history.

> The first known use of the softmax function predates machine learning. The softmax function is in fact borrowed from physics and statistical mechanics, where it is known as the Boltzmann distribution or the Gibbs distribution. It was formulated by the Austrian physicist and philosopher Ludwig Boltzmann in 1868.

> Boltzmann was studying the statistical mechanics of gases in thermal equilibrium. He found that the Boltzmann distribution could describe the probability of finding a system in a certain state, given that state’s energy, and the temperature of the system. His version of the formula was similar to that used in reinforcement learning. Indeed, the parameter τ is called temperature in the field of reinforcement learning as a homage to Boltzmann.

> In 1902 the American physicist and chemist Josiah Willard Gibbs popularized the Boltzmann distribution when he used it to lay the foundation for thermodynamics and his definition of entropy. It also forms the basis of spectroscopy, that is the analysis of materials by looking at the light that they absorb and reflect.

> n 1959 Robert Duncan Luce proposed the use of the softmax function for reinforcement learning in his book Individual Choice Behavior: A Theoretical Analysis. Finally in 1989 John S. Bridle suggested that the argmax in feedforward neural networks should be replaced by softmax because it “preserves the rank order of its input values, and is a differentiable generalisation of the ‘winner-take-all’ operation of picking the maximum value”. In recent years, as neural networks have become widely used, the softmax has become well known thanks to these properties.

src: https://deepai.org/machine-learning-glossary-and-terms/softmax-layer

-------------

## Bridle 1989

## Luce 1959
Page 36

P(i; T) = v(i) / sum_{j in T} v(j)

